# Auto

[![CI](https://github.com/SSHDotCodes/auto/actions/workflows/ci.yml/badge.svg)](https://github.com/SSHDotCodes/auto/actions/workflows/ci.yml)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-22543d.svg)](LICENSE)

![Auto — local permission intelligence](site/assets/mountain.webp)

**Local permission intelligence for coding agents.** Works with Claude Code, Codex, Gemini CLI, Cursor, GitHub Copilot CLI, Pi, OpenCode and Hermes, plus the Claude Agent SDK, LangChain and any custom harness.

[Website](https://auto.ssh.codes) · Models: [auto-0.4b-2](https://huggingface.co/ProCreations/auto-0.4b-2), [auto-200m-2](https://huggingface.co/ProCreations/auto-200m-2) · [Compatibility](docs/compatibility.md) · [Agent integrations](docs/agents.md)

Auto checks a proposed tool call against the user's request and the agent's tool history. A ModernBERT classifier runs on your computer; prompts stay local. Model files download during installation, then inference works offline. Four weight variants are available: **auto-0.4b-2** (395.8M parameters, the default) and the smaller, faster **auto-200m-2** (149.6M). Both read up to 65,536 tokens. The 200M model also comes in int8 (150 MB) and int4 (77 MB), with packed weights on CPU, Apple MPS, NVIDIA CUDA, and AMD ROCm.

## Install

**macOS / Linux** (downloads uv and Python 3.13 if needed):

```sh
curl -fsSL https://auto.ssh.codes/install.sh | sh
```

**Windows PowerShell**:

```powershell
& ([scriptblock]::Create((irm https://auto.ssh.codes/install.ps1)))
```

With no options, the installer connects Auto to every supported agent it finds on your computer. To pick one, add `--agent NAME` (`sh -s -- --agent claude-code`; PowerShell: `-Agent claude-code`) with `claude-code`, `codex`, `gemini`, `cursor`, `copilot`, `pi`, `opencode`, `hermes`, or `all`. Restart the agent after installation. The installer creates an isolated environment, selects a PyTorch build, downloads the pinned model (~0.8 GB), installs the adapter, and warms the local runtime. No sudo or administrator access is needed. For the smaller model add `--model auto-200m-2` (PowerShell: `-Model auto-200m-2`; ~0.3 GB).

Prefer reviewing scripts first? Download [install.sh](scripts/install.sh), [install.ps1](scripts/install.ps1), and [install.py](scripts/install.py), inspect them, and run the appropriate one. The Python installer accepts `--source /path/to/this/repo` for development.

Existing Python environment:

```sh
python -m pip install 'auto-local @ git+https://github.com/SSHDotCodes/auto@v0.3.0'
auto install                        # every agent found; or: auto install claude-code codex --model auto-200m-2
```

For a specific GPU build, use the installer or install the appropriate [PyTorch wheel](https://pytorch.org/get-started/locally/) before this package. See the tested and supported configurations in [compatibility](docs/compatibility.md).

## Integrations

| Agent | `auto install …` | Integration | Auto denies | Auto needs review |
|---|---|---|---|---|
| Claude Code | `claude-code` | `PreToolUse` hook in `~/.claude/settings.json` | Denies the call | Asks you (Claude Code's prompt) |
| Codex CLI | `codex` | `PreToolUse` hook in `~/.codex/hooks.json` | Denies the call | Blocks; the agent asks you to confirm, then retries |
| Gemini CLI | `gemini` | `BeforeTool` hook in `~/.gemini/settings.json` | Denies the call | Blocks; the agent asks you to confirm, then retries |
| Cursor | `cursor` | `preToolUse` hook in `~/.cursor/hooks.json` | Denies the call | Blocks; the agent asks you to confirm, then retries |
| GitHub Copilot CLI | `copilot` | `preToolUse` hook in `~/.copilot/hooks/auto.json` | Denies the call | Asks you (blocks in autopilot) |
| Pi | `pi` | `tool_call` extension | Blocks the call | Prompts in UI; blocks in headless mode |
| OpenCode | `opencode` | `tool.execute.before` plugin | Throws a blocking tool error | Blocks with an actionable message |
| Hermes | `hermes` | `pre_tool_call` plugin | Returns the host's block directive | Requests Hermes's native human approval |
| Claude Agent SDK | — | `claude_agent_hook` (Python) | Denies the call | Asks through the SDK's permission flow |
| LangChain / LangGraph | — | `AutoMiddleware()` | Returns an error tool result | Returns an error tool result |
| Anything else | — | `check()` in Python, `auto score`, or the local HTTP API | Your harness maps the decision | Your harness must require review |

An Auto approval lets the call continue through the host's own permissions: hooks stay silent, so deny rules, sandboxes and permission prompts still apply. Cursor is the exception: it rejects a silent permission hook, so Auto answers `allow` there. Integrations read the user's actual request and the completed tool history, from the host's transcript (Claude Code, Codex) or recorded by Auto's prompt and post-tool hooks (Gemini CLI, Cursor, Copilot CLI). Missing context, invalid responses, timeouts and crashes never become an automatic approval. Per-agent details, tested versions and caveats: [agent integrations](docs/agents.md).

### Frameworks and custom harnesses

```python
from auto_gate.integrations import check

result = check(
    "Clean up the merged branches. Keep main.",  # the user's own words
    {"tool": "bash", "args": {"command": "git push origin --delete exp-7"}},
    history=[{"tool": "bash", "args": {"command": "git branch --merged"}, "result": "exp-7\nmain"}],
)
if result["decision"] != "approve":  # "deny" or "review": don't run it unattended
    ...

# Claude Agent SDK
from claude_agent_sdk import ClaudeAgentOptions, HookMatcher
from auto_gate.integrations import claude_agent_hook

options = ClaudeAgentOptions(hooks={"PreToolUse": [HookMatcher(hooks=[claude_agent_hook])]})

# LangChain v1 agents
from langchain.agents import create_agent
from auto_gate.integrations import AutoMiddleware

agent = create_agent(model, tools, middleware=[AutoMiddleware()])
```

These run in the environment where Auto is installed (`pip install auto-local` from this repository). From other languages, pipe the JSON request below to `auto score`.

## Runtime

```sh
auto doctor                         # actual backend, model revision, local paths
auto start                          # warm or reuse the local process
auto stop                           # unload the model
auto configure --device cpu         # force CPU
auto configure --device mps         # Apple Silicon
auto configure --device cuda        # NVIDIA CUDA or AMD ROCm
auto configure --model auto-200m-2  # smaller, faster model (auto download fetches it)
auto configure --model auto-3b-3    # most accurate; needs a GPU (CUDA or Apple MPS), ~6.2 GB
auto configure --max-tokens 65536   # the model's full context
auto configure --fit-tokens 16384   # size long sessions are fitted to (default: automatic)
auto uninstall claude-code          # remove one integration; retain downloaded model
```

The default device is `auto`: usable CUDA/ROCm, then Apple MPS, then CPU. If the optional FlashAttention kernel cannot load, Auto uses its portable attention on the same GPU; other GPU initialization failures fall back to CPU where possible. `auto doctor` reports the device and attention actually used. No FlashAttention compilation is required. For optional FlashAttention, install this repository with its `flash` extra in the same Python environment, run `auto configure --attention flash` on a compatible NVIDIA system, then `auto download` to fetch the pinned kernel.

### auto-3b-3 (GPU)

auto-3b-3 is a 3.08B-parameter decoder classifier (SmolLM3), the most accurate Auto model. The runtime loads it in BF16 (about 6.2 GB) with PyTorch's fused causal attention, classifies from the last token, and fits long sessions like the other models. Use it with a CUDA or ROCm GPU. On an Apple M-series GPU (MPS) it runs, but slowly: a median of 0.86 s per decision on the v3 benchmark, and about 12 s for sessions fitted to 8k tokens. There, auto-0.4b-3 is the better default. On a CPU, long requests take minutes. Through the runtime on MPS, auto-3b-3 agreed with the reference scores on 299 of 300 benchmark items. To try local weights before they are published, point `AUTO_MODEL_DIR` at a folder containing `config.json`, the tokenizer and `model.safetensors`.

### Quantized models (v0.3.0)

```sh
auto configure --model auto-200m-2-int4   # or auto-200m-2-int8
auto download
auto start
auto doctor
```

For a new installation, add `--model auto-200m-2-int4` to the shell installer, or
`-Model auto-200m-2-int4` in PowerShell. Both quant names work in all installers and `auto install`.
The default remains auto-0.4b-2; upgrading preserves your model choice.

| Variant | Weights download | Published benchmark | False approve / false deny |
|---|---:|---:|---:|
| [auto-200m-2-int8](https://huggingface.co/ProCreations/auto-200m-2-int8) | 150 MB | 2,890/3,000 (96.33%) | 53 / 57 |
| [auto-200m-2-int4](https://huggingface.co/ProCreations/auto-200m-2-int4) | 77 MB | 2,884/3,000 (96.13%) | 60 / 56 |

The runtime reads the pinned `auto-quant-v1` safetensors locally, without executing downloaded Python or
requantizing the weights. Integers stay packed in memory. Only the current linear weight is expanded for a
matrix multiply; the current MLP reuses its two expanded weights across token chunks, then releases them.
Embedding lookups are chunked too. There is no full-model floating-point cache. `auto doctor` reports
`quantization`, compute dtype, device, attention path, and any fallback.

Quantization saves weight storage, not activation storage. All variants use the same exact bounded-memory
attention and keep the complete context. Long-context memory remains linear, while global attention work
still grows quadratically. Portable floating-point matrix multiplies keep these formats usable on every
supported backend; int4 is not promised to outrun int8 or BF16 on every device. See the
[v0.3.0 measurements and limits](docs/verification/v0.3.0/README.md).

### Long contexts

Both models read 65,536 tokens. Agent sessions grow far past that: a day of Claude Code is easily hundreds of thousands of tokens. So when a request is too long, Auto **fits** it rather than giving up. In priority order it keeps:

1. the proposed call;
2. your first and latest messages;
3. the newest completed actions, up to half of the remaining room;
4. your other messages, newest first;
5. older actions that name the same rare identifiers as the call (branch names, paths, hosts, IDs);
6. more of the newest actions.

What it keeps stays in its original order. Step 5 matters because the deciding evidence (say, "branch exp-7 is not merged", 300 actions back) usually names what the call names. On the 413 v3 benchmark items longer than 8,192 tokens, auto-0.4b-3 fitted to 8k scores 91.8% versus 92.5% with the full context (missed harm 7.7% vs 5.1%). Keeping only the newest actions scored 85.5%. When the latest tool output or your first message alone is too long, Auto keeps its start and end with a marked gap. The response reports what was left out (`trimmed`), and the reason says so. Review is returned only if the proposed call and your latest message alone do not fit.

The target size depends on the device, because the hook runs before every tool call and long inputs are slow off-GPU (see the table): 32,768 tokens on CUDA/ROCm and 8,192 on Apple MPS and CPU. Change it with `auto configure --fit-tokens N`, up to `--max-tokens`. `auto configure --fit-context off` restores the strict behaviour: requests over `--max-tokens` get review and no text is dropped.

Where FlashAttention is not available (CPU, Apple MPS, ROCm, older NVIDIA GPUs, Windows), Auto computes ModernBERT's exact attention without ever allocating an N×N score matrix or mask: global layers use PyTorch's fused attention kernels, or query blocks with a fixed memory budget where none applies; sliding-window layers attend each block of queries only to its own window; the per-token MLP runs in chunks. Memory grows linearly with context, so latency, not memory, sets the practical limit. One request through this path (the default; FlashAttention off), extra memory above the loaded model and time (Apple M4 Max in FP32, RTX PRO 6000 in BF16):

| Model | Device | Auto | 16k tokens | 32k tokens | 64k tokens |
|---|---|---|---:|---:|---:|
| auto-0.4b-2 | M4 Max, MPS | v0.1.0 | +1.1 GB · 10 s | +1.8 GB · 69 s | **+12.4 GB · 366 s** |
| auto-0.4b-2 | M4 Max, MPS | v0.2.0 | +1.0 GB · 2.9 s | +1.0 GB · 10 s | +3.1 GB · 42 s |
| auto-0.4b-2 | M4 Max, CPU | v0.1.0 | +1.3 GB · 18 s | +2.5 GB · 114 s | +4.7 GB · 810 s |
| auto-0.4b-2 | M4 Max, CPU | v0.2.0 | +0.9 GB · 12 s | +1.7 GB · 37 s | +2.9 GB · 142 s |
| auto-200m-2 | M4 Max, MPS | v0.2.0 | +1.0 GB · 1.5 s | +1.0 GB · 6.2 s | +2.1 GB · 33 s |
| auto-200m-2 | M4 Max, CPU | v0.2.0 | +0.6 GB · 6.3 s | +1.1 GB · 22 s | +2.2 GB · 80 s |
| auto-0.4b-2 | RTX PRO 6000, CUDA | v0.2.0 | +0.45 GB · 0.12 s | +0.89 GB · 0.31 s | +1.8 GB · 0.87 s |
| auto-200m-2 | RTX PRO 6000, CUDA | v0.2.0 | +0.34 GB · 0.06 s | +0.67 GB · 0.17 s | +1.3 GB · 0.49 s |

MPS figures are Metal driver allocations, which grow in roughly 1 GB heaps. For contrast, stock Transformers attention without FlashAttention on the same RTX PRO 6000 needs +1.0 GB at 8k and +4.0 GB at 16k tokens and grows quadratically from there. With every fused attention kernel disabled, the worst case for a GPU without memory-efficient attention, Auto's peak stayed at 2.4–3.8 GB up to 64k tokens. The full 3,000-item benchmark through this path on CUDA came within one item of the published FlashAttention results for both models.

There is no claim that all GPU generations have the same speed or numerical behavior. Raw measurements: [verification](docs/verification/).

The local service starts on demand, binds only `127.0.0.1` on an available port, requires a per-run bearer token, and rejects browser-origin requests. Its private runtime state stays in the user data directory. It logs no prompts or tool arguments. Set `AUTO_HOME` to override the data directory; host proxies are ignored for local requests.

## Measured model results

Same 3,000-example Approve-or-Deny evaluation, full input lengths, threshold 0.5:

| Model | Accuracy | False approve | False deny |
|---|---:|---:|---:|
| Original auto-0.4b | 89.63% | 6.07% | 14.13% |
| auto-1b-bf16 | 96.60% | 4.14% | 2.75% |
| **auto-0.4b-2** | **96.77%** | **3.14%** | **3.31%** |
| auto-200m-2 | 96.33% | 3.78% | 3.56% |

auto-0.4b-2: independent validation audit **97.88%** (2,595 examples), 16k–64k benchmark slice **94.56%** (239 examples). auto-200m-2: validation audit **98.34%**, 16k–64k slice **94.14%**, at 38% of the parameters. The 0.17-point lead of auto-0.4b-2 over auto-1b is five examples, and its 0.43-point lead over auto-200m-2 is thirteen, not evidence of universal superiority. These are the published BF16/FlashAttention reference measurements. On an RTX PRO 6000, the full benchmark through Auto's portable attention came within one item of both (2,902 and 2,889; see [Long contexts](#long-contexts)); short smoke checks on other devices do not establish identical full-benchmark accuracy everywhere. Full evaluations: [auto-0.4b-2](https://huggingface.co/ProCreations/auto-0.4b-2), [auto-200m-2](https://huggingface.co/ProCreations/auto-200m-2).

## JSON interface

Save this as `request.json`:

```json
{
  "user_request": "Read the project README and summarize setup instructions.",
  "history": [],
  "call": {"tool": "read_file", "args": {"path": "README.md"}},
  "context_complete": true
}
```

```sh
auto score --file request.json
```

Responses contain `decision` (`approve`, `deny`, or `review`), `p_deny`, and `reason`, plus `trimmed` when a long session was fitted. Review responses may omit the probability. The CLI only classifies text; it never executes the proposed call. See [architecture](docs/architecture.md) for protocol and integration details.

## Development and verification

```sh
uv venv --python 3.13
uv pip install -e '.[dev]'
uv run pytest -m 'not model'
node --test tests/*.test.mjs
uv run pytest -m model              # downloads and evaluates real weights
```

CI tests Linux, Windows, and macOS, adapter behavior, installer selection, package builds, and real-model CPU inference. Hardware-specific results are recorded separately; unavailable GPU families are not labeled hardware-tested.

Auto is a probabilistic classifier, not an operating-system sandbox. It cannot inspect hidden script behavior or prove an action safe. Other agent plugins may change arguments after a hook; install Auto after argument-mutating plugins and retain host boundaries. See [security](SECURITY.md), [contributing](CONTRIBUTING.md), and [license](LICENSE).
