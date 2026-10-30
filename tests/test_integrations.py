import asyncio
import json

import pytest

from auto_gate import integrations


@pytest.fixture
def judged(monkeypatch):
    seen = []

    def fake_check(user_request, call, history=(), context_complete=True):
        seen.append({"user_request": user_request, "call": call, "history": list(history)})
        command = json.dumps(call["args"])
        return {"decision": "deny" if "rm -rf" in command else "approve", "reason": "test"}

    monkeypatch.setattr(integrations, "check", fake_check)
    return seen


def langchain_agent(command):
    pytest.importorskip("langchain")
    from langchain.agents import create_agent
    from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
    from langchain_core.messages import AIMessage
    from langchain_core.tools import tool

    ran = []

    @tool
    def shell(command: str) -> str:
        """Run a shell command."""
        ran.append(command)
        return "ok"

    class Model(GenericFakeChatModel):
        def bind_tools(self, tools, **kwargs):
            return self

    replies = iter(
        [
            AIMessage(content="", tool_calls=[{"name": "shell", "args": {"command": "ls"}, "id": "c1"}]),
            AIMessage(content="", tool_calls=[{"name": "shell", "args": {"command": command}, "id": "c2"}]),
            AIMessage(content="done"),
        ]
    )
    agent = create_agent(Model(messages=replies), [shell], middleware=[integrations.AutoMiddleware()])
    return agent, ran


@pytest.mark.parametrize("command,runs", [("git status", True), ("rm -rf ~/project", False)])
def test_langchain_middleware_end_to_end(judged, command, runs):
    agent, ran = langchain_agent(command)
    result = agent.invoke({"messages": [{"role": "user", "content": "Show me the repo status"}]})
    assert ran == (["ls", command] if runs else ["ls"])
    assert judged[1]["user_request"] == "Show me the repo status"
    assert judged[1]["history"] == [{"tool": "shell", "args": {"command": "ls"}, "result": "ok"}]
    if not runs:
        blocked = [m for m in result["messages"] if getattr(m, "tool_call_id", None) == "c2"][0]
        assert blocked.status == "error" and "Blocked by Auto (deny)" in blocked.content


def test_langchain_middleware_async(judged):
    agent, ran = langchain_agent("rm -rf /")
    asyncio.run(agent.ainvoke({"messages": [{"role": "user", "content": "status please"}]}))
    assert ran == ["ls"]


def test_langchain_failure_blocks(monkeypatch):
    def broken(*args, **kwargs):
        raise RuntimeError("runtime gone")

    monkeypatch.setattr(integrations, "check", broken)
    agent, ran = langchain_agent("git status")
    agent.invoke({"messages": [{"role": "user", "content": "status"}]})
    assert ran == []


def test_claude_agent_hook_uses_claude_code_format(tmp_path, monkeypatch):
    transcript = tmp_path / "t.jsonl"
    transcript.write_text(
        json.dumps(
            {"type": "user", "uuid": "u", "parentUuid": None, "message": {"role": "user", "content": "hi"}}
        )
    )
    monkeypatch.setattr("auto_gate.client.score", lambda payload: {"decision": "deny", "reason": "no"})
    event = {
        "hook_event_name": "PreToolUse",
        "transcript_path": str(transcript),
        "tool_name": "Bash",
        "tool_input": {},
    }
    output = asyncio.run(integrations.claude_agent_hook(event, "t1", None))
    assert output["hookSpecificOutput"]["permissionDecision"] == "deny"
