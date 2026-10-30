"""Drop-in guards for agent frameworks and custom harnesses.

Each guard rebuilds the user's request and the completed actions from the framework's own state, scores the proposed
call with the local runtime, and lets only approvals through. Frameworks are optional: import the one you use.

    from auto_gate.integrations import check                       # any harness
    check("Clean up merged branches", {"tool": "bash", "args": {"command": "git branch -d exp"}}, history)

    from auto_gate.integrations import claude_agent_hook            # Claude Agent SDK (Python)
    ClaudeAgentOptions(hooks={"PreToolUse": [HookMatcher(hooks=[claude_agent_hook])]})

    from auto_gate.integrations import AutoMiddleware               # LangChain v1 / LangGraph agents
    create_agent(model, tools, middleware=[AutoMiddleware()])
"""

import asyncio
import json

from .hooks import blocks_text, user_text


def check(user_request, call, history=(), context_complete=True):
    """Score one proposed call. `call` is {"tool", "args"}; `history` holds completed {"tool", "args", "result"}.

    Returns {"decision": "approve" | "deny" | "review", "reason", "p_deny", ...}. Treat anything but "approve" as
    "do not run without a human": every failure is "review", never "approve".
    """
    from .client import score

    return score(
        {
            "user_request": user_request,
            "call": call,
            "history": list(history),
            "context_complete": context_complete,
        }
    )


async def claude_agent_hook(input_data, tool_use_id=None, context=None):
    """Claude Agent SDK PreToolUse hook. Same transcript and output format as the Claude Code hook."""
    from .hooks import run

    output, _ = await asyncio.to_thread(run, "claude-code", json.dumps(input_data))
    return json.loads(output) if output else {}


def langchain_context(messages):
    """User text and completed tool calls from LangChain messages (human / ai with tool_calls / tool)."""
    users, calls, history = [], {}, []
    for message in messages or []:
        kind = getattr(message, "type", None)
        if kind == "human":
            text = user_text(message.content)
            if text:
                users.append(text)
        elif kind == "ai":
            for call in getattr(message, "tool_calls", None) or []:
                calls[call.get("id")] = {"tool": call.get("name") or "tool", "args": call.get("args", {})}
        elif kind == "tool":
            call = calls.pop(getattr(message, "tool_call_id", None), None) or {
                "tool": getattr(message, "name", None) or "tool",
                "args": {},
            }
            history.append({**call, "result": blocks_text(message.content)})
    return users, history


def _langchain_decision(request):
    state = request.state
    messages = state.get("messages") if isinstance(state, dict) else getattr(state, "messages", None)
    users, history = langchain_context(messages)
    call = {"tool": request.tool_call.get("name") or "tool", "args": request.tool_call.get("args", {})}
    if not users:
        return {"decision": "review", "reason": "Auto: review required (no user message in the agent state)"}
    return check("\n\n".join(users), call, history)


def _blocked(request, result):
    from langchain_core.messages import ToolMessage

    return ToolMessage(
        content=f"Blocked by Auto ({result['decision']}): {result.get('reason', '')}. "
        "Do not retry this action without the user's explicit approval.",
        tool_call_id=request.tool_call.get("id") or "",
        name=request.tool_call.get("name"),
        status="error",
    )


def AutoMiddleware():  # noqa: N802 - reads like the class it returns
    """LangChain v1 agent middleware: scores every tool call; deny and review return an error ToolMessage."""
    from langchain.agents.middleware import AgentMiddleware

    class _AutoMiddleware(AgentMiddleware):
        def wrap_tool_call(self, request, handler):
            try:
                result = _langchain_decision(request)
            except Exception:  # noqa: BLE001 - never approve on failure
                result = {"decision": "review", "reason": "Auto could not evaluate this call"}
            return handler(request) if result.get("decision") == "approve" else _blocked(request, result)

        async def awrap_tool_call(self, request, handler):
            try:
                result = await asyncio.to_thread(_langchain_decision, request)
            except Exception:  # noqa: BLE001
                result = {"decision": "review", "reason": "Auto could not evaluate this call"}
            return (
                await handler(request) if result.get("decision") == "approve" else _blocked(request, result)
            )

    return _AutoMiddleware()
