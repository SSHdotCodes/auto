# Agent integrations

Install the agent normally, then run `auto install`. With no name it connects every supported agent it finds (an executable on `PATH` or the agent's config folder). Or name agents: `auto install claude-code codex`, or `all`. The bootstrap installer combines Python and dependency setup with this step. Restart the host agent to load the integration. Auto does not need PyTorch in the agent's own Python environment.

Every integration preserves your other settings and plugins. Before editing an existing settings file, it saves a private timestamped backup next to it (`*.auto-backup-*`). It refuses to touch a file it cannot parse, and `auto uninstall AGENT` removes only Auto's entries.

## Command-hook agents

Claude Code, Codex, Gemini CLI, Cursor and GitHub Copilot CLI run a command before each tool call. Auto registers `"<python>" -m auto_gate hook AGENT --event EVENT --home "<data dir>"` with absolute paths, so the hook works from any shell, virtual environment or working directory.

The hook reads the host's JSON event on stdin. It rebuilds the user's own words and the completed actions (call, arguments, result), scores the proposed call through the local daemon, and prints the host's decision format. Approval prints nothing wherever the host allows that, so the host's own permission rules and prompts still apply. A crash, timeout, unreachable daemon, missing user context or invalid response is answered as review, never approval. On POSIX shells, the Claude Code, Codex and Gemini CLI hooks end with `|| exit 2`, so even a missing interpreter blocks the call rather than skipping the hook (Cursor's uses `failClosed`; Copilot denies on any failure exit). The first call after a restart loads the model, so these hooks get a 180-second timeout.

Long sessions are [fitted to the context budget](../README.md#long-contexts). User text excludes what the host injects into user turns: system reminders, skill and AGENTS.md bodies, environment context, task notifications, and compaction summaries.

### Claude Code

`auto install claude-code` adds one `PreToolUse` entry (matcher `*`) to `~/.claude/settings.json`, or under `CLAUDE_CONFIG_DIR`.

**Context.** Context comes from the session transcript (`transcript_path`):

- User prompts are taken along the active branch back through compactions.
- Completed tool calls come from after the latest compaction.
- Subagent (sidechain) records and the pending call itself are excluded.
- Transcripts can reach hundreds of MB. Auto parses only the newest 16 MB in full and scans older lines just for user prompts. On a 519 MB real transcript the hook spends about 0.35 s reading context.

**Decisions.**

- Deny returns `permissionDecision: "deny"`.
- Review returns `"ask"`, which is Claude Code's own confirmation prompt. In `-p` mode that means a denial.
- Approval returns nothing: Claude Code's permission flow continues ("staying silent doesn't approve it").

**Other agents reading this file.** Cursor and Copilot CLI also run hooks from Claude Code settings files. Auto recognises their payloads there and stays silent; install `cursor` or `copilot` to guard those agents.

The [Claude Agent SDK](#frameworks) uses the same hook through `claude_agent_hook`. Contract: [Claude Code hooks](https://code.claude.com/docs/en/hooks.md#pretooluse-decision-control), checked against Claude Code 2.1.314.

### Codex CLI

`auto install codex` adds a `PreToolUse` entry to `~/.codex/hooks.json`, or under `CODEX_HOME`.

**Trust the hook once.** Codex runs a new or changed hook only after you trust it: open Codex and use `/hooks` to trust Auto's hook. Hooks are enabled by default in current Codex. The hook covers shell commands, `apply_patch` edits, MCP tools and other local tools, but not hosted web search.

**Context.** Context comes from the rollout file (`transcript_path`):

- User messages are kept, minus the AGENTS.md, environment-context and skill blocks Codex injects as user turns.
- Calls and outputs are paired by `call_id`, after the latest compaction.
- Codex calls this file format "not a stable interface". The parser is tested against its current shape and returns review rather than guessing when nothing usable is found.

**Decisions.** A Codex `PreToolUse` hook can deny but cannot ask. So deny and review both block. The review message tells the agent to get your explicit confirmation; your reply becomes part of the context for the retry. Approval returns nothing, and Codex's own approval policy still applies.

Contract: [Codex hooks](https://learn.chatgpt.com/docs/hooks); [`pre_tool_use.rs` at rust-v0.166.0](https://github.com/openai/codex/blob/rust-v0.166.0/codex-rs/hooks/src/events/pre_tool_use.rs).

### Gemini CLI

`auto install gemini` adds three entries to the `hooks` key of `~/.gemini/settings.json`:

- `BeforeAgent` records your prompt.
- `AfterTool` records each completed call and its result.
- `BeforeTool` (matcher `*`) decides.

The recorded context lives in a private per-session file in Auto's data folder. It is pruned after seven days. Results longer than 200,000 characters keep their start and end.

**Decisions.** Deny and review return `{"decision": "deny", "reason": …}`. The reason reaches the agent as a tool error, and for review it asks the agent to get your confirmation. Approval returns nothing: Gemini's policy engine still runs.

Contract: [Gemini CLI hooks reference at v0.66.0](https://github.com/google-gemini/gemini-cli/blob/v0.66.0/docs/hooks/reference.md).

### Cursor

`auto install cursor` adds three entries to `~/.cursor/hooks.json`:

- `beforeSubmitPrompt` records your prompt (it always answers `continue`).
- `postToolUse` records results.
- `preToolUse` decides, with `failClosed: true`.

Context is keyed by `conversation_id`.

**Decisions.** Cursor treats empty output from a permission hook as invalid, so approval is an explicit `{"permission": "allow"}`. Cursor does not document whether `allow` also skips its own approval prompts. `preToolUse` does not enforce `ask` today, so review is a deny that asks the agent to get your confirmation. Hooks in the Cursor CLI (`cursor-agent`) have had gaps; check Cursor's release notes for your version.

Contract: [Cursor hooks](https://cursor.com/docs/hooks), checked against Cursor 3.27 and CLI 2026.10.22.

### GitHub Copilot CLI

`auto install copilot` writes its own file, `~/.copilot/hooks/auto.json` (or under `COPILOT_HOME`), with three hooks:

- `userPromptSubmitted` records your prompt.
- `postToolUse` records results.
- `preToolUse` decides.

**Decisions.**

- Deny returns `permissionDecision: "deny"`.
- Review returns `"ask"`, which Copilot turns into a deny in autopilot mode and in the cloud agent.
- Approval returns nothing, so Copilot's default behaviour applies.
- Copilot denies on any `preToolUse` failure exit. A hook timeout continues the call, which is why the timeout is 180 seconds.

Contract: [Copilot hooks reference](https://docs.github.com/en/copilot/reference/hooks-reference), checked against Copilot CLI 1.0.105.

### Not yet integrated

- **Factory Droid** and **Devin** (formerly Windsurf Cascade) have similar hooks, but their transcript formats are not documented; Devin also runs Claude Code's settings hooks, where Auto stays out of the way.
- **Cline**'s file hooks are no longer documented, and its `cancel` output stops the whole task rather than one call.

These agents can still call `auto score` from their own hook scripts.

## Plugin agents

### Pi

The installer writes `~/.pi/agent/extensions/zz-auto.ts`, or under `PI_CODING_AGENT_DIR`. The generated extension imports the installed adapter by an absolute file URI, including paths with spaces and Unicode.

The adapter reads the current session branch at each `tool_call`. User-role text goes into the user request; completed tool calls and their results go into agent history. A denial returns `{block: true, reason}`. A review request offers a one-call confirmation through `ctx.ui.confirm`; headless sessions block. Approval returns normally so the rest of the host pipeline still runs.

Source contract checked against [earendil-works/pi](https://github.com/earendil-works/pi/blob/6160683a4a8012f0d1cd30c145df18b4ca6f5176/packages/coding-agent/docs/extensions.md) at `6160683a4a8012f0d1cd30c145df18b4ca6f5176`.

### OpenCode

The installer writes `~/.config/opencode/plugins/zz-auto.ts`, respecting `XDG_CONFIG_HOME`. The async plugin hooks `tool.execute.before`, obtains the relevant session transcript through the host SDK, and evaluates the proposed `output.args`. Completed tools become history; synthetic user text is excluded from user authorization.

A deny or review throws a blocking tool error with a useful message. The plugin does not override `permission.ask` or auto-answer existing permission prompts. If a call needs review, inspect the request and adjust your workflow; there is intentionally no persistent "allow all" escape hatch in this plugin.

Source contract checked against OpenCode **1.18.30**, commit [830d5eb](https://github.com/anomalyco/opencode/blob/830d5eb5354874105cc31599635a80c1662609e8/packages/plugin/src/index.ts). [Official plugin documentation](https://opencode.ai/docs/plugins/).

### Hermes

The installer creates `~/.hermes/plugins/auto/{plugin.yaml,__init__.py}`, respecting `HERMES_HOME`, and adds `auto` to `plugins.enabled` in `config.yaml`. Existing settings and other enabled plugins remain. Before rewriting YAML, the installer saves a private timestamped backup; comments may be reformatted by YAML serialization. Uninstall removes only Auto's entry and its owned loader files.

`pre_llm_call` captures the original user request and conversation history; `post_tool_call` appends completed actions; state is isolated by session/task and cleared on `on_session_end`. `pre_tool_call` scores the proposed action. A deny returns `action: block`. Review returns Hermes's `action: approve`, which **requests human approval** in Hermes terminology; it does not grant it. Auto approval returns `None` and preserves Hermes's other protections.

The subprocess deadline is 20 seconds, below Hermes's default hook timeout. A slow or unavailable runtime requests native approval. Headless behavior then follows Hermes's native approval policy. Source contract checked against Hermes **0.21.1**, commit [0e9fc2c](https://github.com/NousResearch/hermes-agent/blob/0e9fc2cc152b4a4d9fd736f107412ace2a0c2555/hermes_cli/plugins.py). [Official hook documentation](https://hermes-agent.nousresearch.com/docs/user-guide/features/hooks/).

## Frameworks

`auto_gate.integrations` ships drop-in guards for code that runs where Auto is installed:

- **`check(user_request, call, history=())`** scores one call for any harness. It returns the JSON response below. Run the call only on `"approve"`.
- **`claude_agent_hook`** is a Claude Agent SDK `PreToolUse` hook: `ClaudeAgentOptions(hooks={"PreToolUse": [HookMatcher(hooks=[claude_agent_hook])]})`. It reads the SDK session transcript exactly like the Claude Code hook.
- **`AutoMiddleware()`** is LangChain v1 agent middleware (`create_agent(..., middleware=[AutoMiddleware()])`, sync and async). It reads human messages, AI tool calls and tool results from the agent state. A deny or review returns an error `ToolMessage` instead of running the tool; failures block. Tested with langchain 1.4.7 / langgraph 1.2.18.

The OpenAI Agents SDK is not covered. Its tool guardrails see the run's original input but not the tool calls and results from the current run, which are the evidence Auto needs for injected instructions. Use `check()` from your own tool wrapper with the run's full item list.

## Boundaries common to every adapter

Auto only sees the context supplied by the host and evaluates tools routed through the host's supported hook. Subagents need the integration in their own sessions. External programs or actions that bypass those hooks are not intercepted. A skill's name is text; its actual tool calls are scored when the agent executes them. Auto cannot open a hidden script to discover its behavior.

Keep user instructions separate from retrieved content. Preserve enough history to assess authorization, and report missing context. No adapter truncates a long request just to obtain an answer. The runtime returns review instead.

Other plugins may run before or after Auto. The `zz-` filename helps place it after many local argument-mutating plugins, but does not guarantee ordering across all host/plugin sources. In Hermes, multiple policy plugins can also affect directive priority. Keep the agent's own permissions and OS boundaries enabled. Updates to upstream hook contracts require rerunning the contract tests; exact tested revisions are recorded above.

## Remove or upgrade

Run `auto uninstall AGENT` to remove the integration. Run `auto stop` to unload the process. The model cache remains for reuse. Remove the Auto data/install directories shown by `auto doctor` only when you intentionally want to reclaim them. Never remove another agent's entire config folder.

Rerun the bootstrap installer to update the pinned release and rewrite owned loaders. It refuses to overwrite unrelated code at an adapter path. Use `--source /path/to/checkout` with the Python installer for development. If the installation is moved to a new directory, rerun `auto install AGENT` so absolute loader paths are refreshed.
