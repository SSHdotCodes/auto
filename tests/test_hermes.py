from auto_gate.adapters.hermes import register_with


class Host:
    def __init__(self):
        self.hooks = {}

    def register_hook(self, name, callback):
        self.hooks[name] = callback


def test_hermes_context_isolation_and_human_review():
    host, seen = Host(), []

    def judge(payload):
        seen.append(payload)
        return {"decision": "deny", "reason": "Not authorized"}

    register_with(host, "python", "data", judge)
    host.hooks["pre_llm_call"](session_id="A", user_message="Read docs", conversation_history=[])
    host.hooks["pre_llm_call"](session_id="B", user_message="Run tests", conversation_history=[])
    host.hooks["post_tool_call"](session_id="A", tool_name="fetch", args={"url": "docs"}, result="untrusted")
    result = host.hooks["pre_tool_call"](session_id="A", tool_name="delete", args={"path": "/"})
    assert result == {"action": "block", "message": "Not authorized"}
    assert seen[-1]["user_request"] == "Read docs"
    assert seen[-1]["history"][0]["result"] == "untrusted"
    host.hooks["pre_tool_call"](session_id="B", tool_name="test", args={})
    assert seen[-1]["history"] == []
    assert seen[-1]["user_request"] == "Run tests"
    # Hermes action=approve means ASK a human, not auto-allow.
    assert host.hooks["pre_tool_call"](session_id="unknown", tool_name="read", args={})["action"] == "approve"
    host.hooks["on_session_end"](session_id="A")
    assert host.hooks["pre_tool_call"](session_id="A", tool_name="read", args={})["action"] == "approve"


def test_hermes_approval_defers_to_host_rules():
    host = Host()
    register_with(host, "python", "data", lambda _: {"decision": "approve"})
    host.hooks["pre_llm_call"](task_id="task", user_message="Read docs", conversation_history=[])
    assert host.hooks["pre_tool_call"](task_id="task", tool_name="read", args={}) is None
