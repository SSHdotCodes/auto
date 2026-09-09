"""Load the actual installed Auto plugin through pinned Hermes discovery and dispatch."""

import json
import os
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from auto_gate.install import install_agent

source = Path(os.environ["HERMES_SOURCE"]).resolve()
sys.path.insert(0, str(source))
with tempfile.TemporaryDirectory(prefix="auto hermes ü ") as folder:
    root = Path(folder)
    os.environ.update(
        HERMES_HOME=str(root / "hermes"),
        AUTO_HOME=str(root / "data"),
        HERMES_BUNDLED_PLUGINS=str(root / "no-bundled-plugins"),
        HERMES_SAFE_MODE="0",
    )
    install_agent("hermes", root / "hermes")
    from hermes_cli.plugins import get_plugin_manager, get_pre_tool_call_directive

    manager = get_plugin_manager()
    manager.discover_and_load()
    assert "auto" in manager._plugins
    assert manager._plugins["auto"].enabled
    for hook in ["pre_llm_call", "pre_tool_call", "post_tool_call", "on_session_end"]:
        assert manager._hooks.get(hook), hook
    manager.invoke_hook(
        "pre_llm_call",
        session_id="contract-session",
        user_message="Read the README.",
        conversation_history=[],
    )
    for expected, action in [("approve", None), ("deny", "block"), ("review", "approve")]:
        # Only replace the external inference subprocess. Discovery, validation, and dispatch are real Hermes.
        response = SimpleNamespace(stdout=json.dumps({"decision": expected, "reason": "Contract fixture"}))
        with patch("subprocess.run", return_value=response):
            directive, message = get_pre_tool_call_directive(
                "read", {"path": "README.md"}, session_id="contract-session"
            )
        assert directive == action, (expected, directive, message)
    manager.invoke_hook("on_session_end", session_id="contract-session")
    directive, _ = get_pre_tool_call_directive("read", {}, session_id="contract-session")
    assert directive == "approve", "Missing context must request native human approval"
    manager.unload("auto")
    print(
        "Hermes real discovery, all four hooks, deny/approve/review directives, and session cleanup passed."
    )
