"""Fit a long session into the classifier's context instead of refusing it (`auto configure --fit-context`).

Priority, highest first: the proposed call; the first and the latest user paragraphs; the newest completed actions
(guaranteed up to half of the remaining room); the other user paragraphs, newest first; older actions. Kept user
paragraphs and actions stay contiguous and in order, so the model never sees a gap in the middle of what it keeps.
If the call plus the first and latest user paragraphs cannot fit, nothing is trimmed and the caller returns review.

The same policy runs twice: by characters in the client, so huge sessions stay under the local request-size limit,
and by exact tokens in the daemon. Standard library only.
"""

import json

SEPARATOR = "\n\n"  # adapters join user messages with a blank line; paragraphs are the trimming unit
ELISION = "\n[... {} characters omitted by Auto to fit its context budget ...]\n"


def text(value):
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def action_text(action):
    return f"[0000] {action.get('tool', '')}({text(action.get('args', {}))})\n-> {text(action.get('result', ''))}"


def shorten(value, keep_chars):
    omitted = len(value) - keep_chars
    if omitted <= 0:
        return value
    head = keep_chars * 2 // 3
    tail = keep_chars - head
    return value[:head] + ELISION.format(omitted) + (value[-tail:] if tail else "")


def elide(action, keep_chars):
    return {**action, "result": shorten(text(action.get("result", "")), keep_chars)}


def fit(payload, cost, budget, base, user_cap=None, max_actions=None):
    """Return (payload, trimmed) where trimmed counts what was left out, or (None, None) if it cannot fit.

    `cost(str)` measures a piece of text, `base` is the cost of everything except user text and history (the call and
    section headers), `budget` the total allowed. `user_cap`/`max_actions` bound the result for schema limits.
    """
    paragraphs = payload["user_request"].split(SEPARATOR)
    latest, first, middle = paragraphs[-1], paragraphs[:-1][:1], paragraphs[1:-1]
    actions = list(payload.get("history") or [])
    sep, newline = cost(SEPARATOR), cost("\n")
    room, user_chars = budget - base - cost(latest), len(latest)
    if room < 0 or (user_cap is not None and user_chars > user_cap):
        return None, None
    shortened = False
    if first:
        spare_chars = None if user_cap is None else (user_cap - user_chars) // 2
        size = cost(first[0]) + sep
        if size > room // 2 or (spare_chars is not None and len(first[0]) > spare_chars):
            # The opening message would crowd out everything else (say, a pasted log): keep its start and end.
            keep = int(len(first[0]) * (room // 4) / max(1, size)) - len(ELISION)
            keep = keep if spare_chars is None else min(keep, spare_chars - len(ELISION))
            first, shortened = [shorten(first[0], max(0, keep))], True
        room -= cost(first[0]) + sep
        user_chars += len(first[0]) + len(SEPARATOR)
    costs = {}

    def action_cost(i):
        if i not in costs:
            costs[i] = cost(action_text(actions[i])) + newline
        return costs[i]

    # Newest actions first, up to half the room: recent provenance (what the agent just read) matters most.
    reserved = 0
    for i in reversed(range(len(actions))):
        if reserved + action_cost(i) > room // 2:
            break
        reserved += action_cost(i)
    user_room = room - reserved
    kept_middle, used = [], 0
    for paragraph in reversed(middle):
        size = cost(paragraph) + sep
        user_chars += len(paragraph) + len(SEPARATOR)
        if used + size > user_room or (user_cap is not None and user_chars > user_cap):
            break
        kept_middle.insert(0, paragraph)
        used += size
    history_room = room - used
    kept, used_history, elided = [], 0, False
    for i in reversed(range(len(actions))):
        if max_actions is not None and len(kept) >= max_actions:
            break
        if used_history + action_cost(i) > history_room:
            spare = history_room - used_history - cost(action_text({**actions[i], "result": ""})) - 32
            if not kept and spare > 64:
                # The newest action alone is too long (say, a huge file read): keep its start and end.
                result = text(actions[i].get("result", ""))
                keep_chars = max(0, int(len(result) * spare / max(1, action_cost(i))) - 80)
                kept.insert(0, elide(actions[i], keep_chars))
                elided = True
            break
        kept.insert(0, actions[i])
        used_history += action_cost(i)
    fitted = {
        **payload,
        "user_request": SEPARATOR.join([*first, *kept_middle, latest]),
        "history": kept,
    }
    trimmed = {
        "actions": len(actions) - len(kept),
        "user_paragraphs": len(middle) - len(kept_middle),
        "elided_result": elided,
        "elided_first_message": shortened,
    }
    return fitted, trimmed


def prefit(payload, limit, user_cap, max_actions):
    """Character-level pass for the client: keep the JSON request under `limit` bytes and within schema limits."""

    def size(value):
        return len(json.dumps(value, ensure_ascii=False, allow_nan=False).encode("utf-8"))

    try:
        history = payload.get("history") or []
        if (
            size(payload) <= limit
            and len(payload["user_request"]) <= user_cap
            and len(history) <= max_actions
        ):
            return payload, None
        base = size({**payload, "user_request": "", "history": []})
        budget = limit
        for _ in range(4):
            fitted, trimmed = fit(payload, size, budget, base, user_cap=user_cap, max_actions=max_actions)
            if fitted is None:
                break
            if size(fitted) <= limit:
                return fitted, trimmed
            budget = budget * 3 // 4
    except (AttributeError, KeyError, TypeError, ValueError):
        pass  # malformed input: schema validation rejects it and the caller returns review
    return payload, None


def merge(*parts):
    """Combine trimmed counts from the client and daemon passes; None when nothing was trimmed."""
    total = {"actions": 0, "user_paragraphs": 0, "elided_result": False, "elided_first_message": False}
    for part in parts:
        for key, value in (part or {}).items():
            total[key] = (total[key] or value) if isinstance(value, bool) else total[key] + value
    return total if any(total.values()) else None


def describe(trimmed):
    if not trimmed:
        return ""
    parts = []
    if trimmed.get("actions"):
        parts.append(f"{trimmed['actions']} older actions")
    if trimmed.get("user_paragraphs"):
        parts.append(f"{trimmed['user_paragraphs']} earlier user paragraphs")
    if trimmed.get("elided_first_message"):
        parts.append("the middle of the first user message")
    if trimmed.get("elided_result"):
        parts.append("the middle of the latest tool output")
    return "; context fitted, omitted " + ", ".join(parts) if parts else ""
