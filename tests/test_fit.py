import json

from auto_gate.fit import action_text, describe, fit, merge, prefit


def actions(n, size=10):
    return [{"tool": "bash", "args": {"n": i}, "result": f"{i}:" + "x" * size} for i in range(n)]


def request(paragraphs, history):
    return {"user_request": "\n\n".join(paragraphs), "call": {"tool": "bash", "args": {}}, "history": history}


def test_fits_untouched_when_under_budget():
    payload = request(["first", "latest"], actions(3))
    fitted, trimmed = fit(payload, len, 10_000, 0)
    assert fitted == payload and merge(trimmed) is None


def test_keeps_first_latest_and_newest_actions_contiguously():
    paragraphs = ["first"] + [f"middle {i} " + "m" * 40 for i in range(20)] + ["latest"]
    payload = request(paragraphs, actions(50))
    fitted, trimmed = fit(payload, len, 1_000, 0)
    kept = fitted["user_request"].split("\n\n")
    assert kept[0] == "first" and kept[-1] == "latest"
    assert (
        kept[1:-1] == paragraphs[len(paragraphs) - len(kept) + 1 : -1]
    )  # newest middle paragraphs, in order
    history = fitted["history"]
    assert history and history == payload["history"][-len(history) :]  # newest actions, contiguous
    assert trimmed["actions"] == 50 - len(history) and trimmed["user_paragraphs"] == 20 - (len(kept) - 2)
    used = len(fitted["user_request"]) + sum(len(action_text(a)) + 1 for a in history)
    assert used <= 1_000


def test_history_keeps_up_to_half_the_room_against_long_user_text():
    paragraphs = ["first"] + ["p" * 100] * 200 + ["latest"]
    payload = request(paragraphs, actions(100))
    fitted, _ = fit(payload, len, 4_000, 0)
    history_size = sum(len(action_text(a)) + 1 for a in fitted["history"])
    assert history_size >= 4_000 // 2 - 2 * len(action_text(payload["history"][0]))


def test_essentials_that_cannot_fit_return_none():
    payload = request(["first", "l" * 500], actions(3))  # the latest message itself is over budget
    assert fit(payload, len, 100, 0) == (None, None)


def test_oversized_newest_result_is_elided_not_dropped():
    history = actions(3) + [
        {"tool": "read_file", "args": {"path": "big"}, "result": "HEAD" + "z" * 20_000 + "TAIL"}
    ]
    fitted, trimmed = fit(request(["first", "latest"], history), len, 2_000, 0)
    assert len(fitted["history"]) == 1 and trimmed["elided_result"]
    result = fitted["history"][0]["result"]
    assert result.startswith("HEAD") and result.endswith("TAIL") and "omitted by Auto" in result
    assert "fitted" in describe(merge(trimmed))


def test_prefit_respects_byte_and_schema_limits():
    payload = request(["first"] + ["u" * 1000] * 600 + ["latest"], actions(3000, size=1000))
    fitted, trimmed = prefit(payload, 1_500_000, user_cap=480_000, max_actions=2048)
    assert len(json.dumps(fitted, ensure_ascii=False).encode()) <= 1_500_000
    assert len(fitted["user_request"]) <= 480_000 and len(fitted["history"]) <= 2048
    assert fitted["history"][-1] == payload["history"][-1] and trimmed["actions"] > 0


def test_prefit_leaves_small_and_malformed_input_alone():
    small = request(["hi"], actions(2))
    assert prefit(small, 1_500_000, 480_000, 2048) == (small, None)
    assert prefit({"call": {}}, 10, 10, 10) == ({"call": {}}, None)
    assert prefit(["not", "a", "dict"], 10, 10, 10) == (["not", "a", "dict"], None)


def stub_classifier(fit_context, max_tokens=600):
    import threading

    import torch

    from auto_gate.config import Settings
    from auto_gate.model import Classifier

    class Tokenizer:  # about four characters per token, plus two special tokens
        def __call__(self, text, return_tensors=None, truncation=False, add_special_tokens=True):
            ids = list(range(-(-len(text) // 4) + (2 if add_special_tokens else 0)))
            return {"input_ids": torch.tensor([ids]) if return_tensors == "pt" else ids}

    classifier = Classifier.__new__(Classifier)
    classifier.settings = Settings(max_tokens=max_tokens, fit_context=fit_context, fit_tokens=512)
    classifier.tokenizer, classifier.lock, classifier.device = Tokenizer(), threading.Lock(), "cpu"
    classifier._forward = lambda ids: torch.tensor([[2.0, -2.0]])
    return classifier


def test_daemon_fits_long_sessions_or_reviews_when_disabled():
    from auto_gate.schema import ScoreRequest

    long = ScoreRequest(**request(["first"] + ["middle"] * 300 + ["latest"], actions(200, size=30)))
    fitted = stub_classifier(True).score(long)
    assert fitted.decision == "approve" and fitted.tokens <= 512
    assert fitted.trimmed["actions"] > 0 and "context fitted" in fitted.reason
    assert stub_classifier(False).score(long).decision == "review"
    huge_latest = ScoreRequest(**request(["first", "x" * 4000], []))  # 1,000 tokens: over the 512 target
    assert stub_classifier(True).score(huge_latest).decision == "review"


def test_oversized_first_message_is_shortened_latest_kept_whole():
    paragraphs = ["START" + "f" * 50_000 + "END", "middle", "latest message"]
    fitted, trimmed = fit(request(paragraphs, actions(5)), len, 4_000, 0)
    first, *_, latest = fitted["user_request"].split("\n\n")
    assert latest == "latest message" and trimmed["elided_first_message"]
    assert first.startswith("START") and first.endswith("END") and len(first) < 1_200
    assert fitted["history"]  # history still has room


def test_old_evidence_naming_the_call_target_is_kept():
    history = actions(200, size=40)
    history[30] = {"tool": "git", "args": "branch -r --merged", "result": "origin/exp-sweep-07 (not merged)"}
    payload = request(["Delete the merged experiment branches."], history)
    payload["call"] = {"tool": "bash", "args": {"command": "git push origin --delete exp-sweep-07"}}
    fitted, trimmed = fit(payload, len, 3_000, 0)
    kept = fitted["history"]
    assert history[30] in kept and kept[-1] == history[-1]  # the deciding old action and the newest
    assert kept == sorted(kept, key=history.index)  # original order
    assert trimmed["actions"] == 200 - len(kept)
