"""Hermes adapter: only standard-library imports, so Hermes need not install torch."""

import json
import os
import subprocess
import threading


def content_text(content):
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(p.get("text", "") for p in content if p.get("type") == "text")
    return ""


def transcript(history, user_message):
    users, actions, calls = [], [], {}
    for message in history:
        if message.get("role") == "user":
            value = content_text(message.get("content"))
            if value:
                users.append(value)
        for call in message.get("tool_calls", []) or []:
            function = call.get("function", {})
            args = function.get("arguments", {})
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except ValueError:
                    pass
            calls[call.get("id")] = {"tool": function.get("name", "tool"), "args": args}
        if message.get("role") == "tool":
            call = calls.get(message.get("tool_call_id"), {"tool": message.get("name", "tool"), "args": {}})
            actions.append({**call, "result": content_text(message.get("content"))})
    if user_message and (not users or users[-1] != user_message):
        users.append(user_message)
    return {"user_request": "\n\n".join(users), "history": actions, "context_complete": bool(users)}


def register_with(ctx, python, auto_home, judge=None):
    states = {}
    lock = threading.RLock()

    def infer(payload):
        try:
            response = subprocess.run(
                [python, "-m", "auto_gate", "bridge"],
                input=json.dumps(payload),
                capture_output=True,
                text=True,
                encoding="utf-8",
                env={**os.environ, "AUTO_HOME": auto_home},
                timeout=20,
                check=True,
            )
            result = json.loads(response.stdout)
            if result.get("decision") not in {"approve", "deny", "review"}:
                raise ValueError("Invalid result")
            return result
        except Exception:
            return {"decision": "review", "reason": "Auto unavailable; Hermes approval required"}

    judge = judge or infer

    def key(session_id="", task_id="", **kwargs):
        return session_id or task_id

    def before_turn(user_message="", conversation_history=None, **kwargs):
        identity = key(**kwargs)
        if identity:
            with lock:
                states[identity] = transcript(conversation_history or [], user_message)

    def after_tool(tool_name, args=None, result="", **kwargs):
        with lock:
            state = states.get(key(**kwargs))
            if state is not None:
                state["history"].append({"tool": tool_name, "args": args or {}, "result": result})

    def before_tool(tool_name, args=None, **kwargs):
        try:
            with lock:
                state = states.get(key(**kwargs))
                if state is None:
                    return {
                        "action": "approve",
                        "message": "Auto has no user context; human approval required",
                    }
                payload = {
                    **state,
                    "history": list(state["history"]),
                    "call": {"tool": tool_name, "args": args or {}},
                }
            result = judge(payload)
            if result["decision"] == "approve":
                return None
            if result["decision"] == "review":
                # In Hermes, action=approve REQUESTS human approval; it does not grant permission.
                return {"action": "approve", "message": result["reason"]}
            return {"action": "block", "message": result["reason"]}
        except Exception:
            return {"action": "block", "message": "Auto adapter failed; call blocked"}

    def end_session(**kwargs):
        with lock:
            states.pop(key(**kwargs), None)

    ctx.register_hook("pre_llm_call", before_turn)
    ctx.register_hook("pre_tool_call", before_tool)
    ctx.register_hook("post_tool_call", after_tool)
    ctx.register_hook("on_session_end", end_session)
