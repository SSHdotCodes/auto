# Agent integrations

Install the agent normally, then run `auto install pi`, `auto install opencode`, or `auto install hermes`. The bootstrap installer combines Python/dependency setup with this step. `all` installs all three. Restart the host agent to load the integration. Auto does not require the agent's Python environment to contain PyTorch.

## Pi

The installer writes `~/.pi/agent/extensions/zz-auto.ts`, or under `PI_CODING_AGENT_DIR`. The generated extension imports the installed adapter by an absolute file URI, including paths with spaces and Unicode.

The adapter reads the current session branch at each `tool_call`. User-role text goes into the user request; completed tool calls and their results go into agent history. A denial returns `{block: true, reason}`. A review request offers a one-call confirmation through `ctx.ui.confirm`; headless sessions block. Approval returns normally so the rest of the host pipeline still runs.

Source contract checked against [earendil-works/pi](https://github.com/earendil-works/pi/blob/6160683a4a8012f0d1cd30c145df18b4ca6f5176/packages/coding-agent/docs/extensions.md) at `6160683a4a8012f0d1cd30c145df18b4ca6f5176`.

## OpenCode

The installer writes `~/.config/opencode/plugins/zz-auto.ts`, respecting `XDG_CONFIG_HOME`. The async plugin hooks `tool.execute.before`, obtains the relevant session transcript through the host SDK, and evaluates the proposed `output.args`. Completed tools become history; synthetic user text is excluded from user authorization.

A deny or review throws a blocking tool error with a useful message. The plugin does not override `permission.ask` or auto-answer existing permission prompts. If a call needs review, inspect the request and adjust your workflow; there is intentionally no persistent "allow all" escape hatch in this plugin.

Source contract checked against OpenCode **1.18.30**, commit [830d5eb](https://github.com/anomalyco/opencode/blob/830d5eb5354874105cc31599635a80c1662609e8/packages/plugin/src/index.ts). [Official plugin documentation](https://opencode.ai/docs/plugins/).

## Hermes

The installer creates `~/.hermes/plugins/auto/{plugin.yaml,__init__.py}`, respecting `HERMES_HOME`, and adds `auto` to `plugins.enabled` in `config.yaml`. Existing settings and other enabled plugins remain. Before rewriting YAML, the installer saves a private timestamped backup; comments may be reformatted by YAML serialization. Uninstall removes only Auto's entry and its owned loader files.

`pre_llm_call` captures the original user request and conversation history; `post_tool_call` appends completed actions; state is isolated by session/task and cleared on `on_session_end`. `pre_tool_call` scores the proposed action. A deny returns `action: block`. Review returns Hermes's `action: approve`, which **requests human approval** in Hermes terminology; it does not grant it. Auto approval returns `None` and preserves Hermes's other protections.

The subprocess deadline is 20 seconds, below Hermes's default hook timeout. A slow or unavailable runtime requests native approval. Headless behavior then follows Hermes's native approval policy. Source contract checked against Hermes **0.21.1**, commit [0e9fc2c](https://github.com/NousResearch/hermes-agent/blob/0e9fc2cc152b4a4d9fd736f107412ace2a0c2555/hermes_cli/plugins.py). [Official hook documentation](https://hermes-agent.nousresearch.com/docs/user-guide/features/hooks/).

## Boundaries common to every adapter

Auto only sees the context supplied by the host and evaluates tools routed through the host's supported hook. Subagents need the integration in their own sessions. External programs or actions that bypass those hooks are not intercepted. A skill's name is text; its actual tool calls are scored when the agent executes them. Auto cannot open a hidden script to discover its behavior.

Keep user instructions separate from retrieved content. Preserve enough history to assess authorization, and report missing context. No adapter truncates a long request just to obtain an answer. The runtime returns review instead.

Other plugins may run before or after Auto. The `zz-` filename helps place it after many local argument-mutating plugins, but does not guarantee ordering across all host/plugin sources. In Hermes, multiple policy plugins can also affect directive priority. Keep the agent's own permissions and OS boundaries enabled. Updates to upstream hook contracts require rerunning the contract tests; exact tested revisions are recorded above.

## Remove or upgrade

Run `auto uninstall AGENT` to remove the integration. Run `auto stop` to unload the process. The model cache remains for reuse. Remove the Auto data/install directories shown by `auto doctor` only when you intentionally want to reclaim them. Never remove another agent's entire config folder.

Rerun the bootstrap installer to update the pinned release and rewrite owned loaders. It refuses to overwrite unrelated code at an adapter path. Use `--source /path/to/checkout` with the Python installer for development. If the installation is moved to a new directory, rerun `auto install AGENT` so absolute loader paths are refreshed.
