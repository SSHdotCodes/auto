import json
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class Call(BaseModel):
    model_config = ConfigDict(extra="forbid")
    tool: str = Field(min_length=1, max_length=512)
    args: Any = Field(default_factory=dict)


class History(Call):
    result: Any = ""


class ScoreRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    user_request: str = Field(min_length=1, max_length=500_000)
    call: Call
    history: list[History] = Field(default_factory=list, max_length=2048)
    context_complete: bool = True


class Decision(BaseModel):
    decision: Literal["approve", "deny", "review"]
    p_deny: float | None = None
    reason: str
    tokens: int | None = None
    device: str | None = None
    model: str | None = None
    revision: str | None = None


def text(value: Any) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def serialize(request: ScoreRequest) -> str:
    """Preserve the trained section order. No truncation, summaries, or synthetic user consent."""
    parts = [
        "### PROPOSED TOOL CALL",
        f"tool: {request.call.tool}",
        f"args: {text(request.call.args)}",
        "",
        "### USER REQUEST",
        request.user_request,
        "",
        "### AGENT HISTORY",
    ]
    if not request.history:
        parts.append("(no prior actions)")
    for i, action in enumerate(request.history, 1):
        parts.append(f"[{i}] {action.tool}({text(action.args)})\n-> {text(action.result)}")
    return "\n".join(parts)
