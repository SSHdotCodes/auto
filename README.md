# Auto

[![CI](https://github.com/SSHDotCodes/auto/actions/workflows/ci.yml/badge.svg)](https://github.com/SSHDotCodes/auto/actions/workflows/ci.yml)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-22543d.svg)](LICENSE)

![Auto — local permission intelligence](site/assets/mountain.webp)

**Local permission intelligence for coding agents.**

[Website](https://auto.ssh.codes) · [Model](https://huggingface.co/ProCreations/auto-0.4b-2) · [Compatibility](docs/compatibility.md) · [Agent integrations](docs/agents.md)

Auto checks a proposed tool call against the user's request and the agent's tool history. A 395.8M-parameter ModernBERT classifier runs on your computer; prompts stay local. Model files download during installation, then inference works offline.

## Install

**macOS / Linux** (downloads uv and Python 3.13 if needed):

```sh
curl -fsSL https://auto.ssh.codes/install.sh | sh -s -- --agent pi
```

**Windows PowerShell**:

```powershell
& ([scriptblock]::Create((irm https://auto.ssh.codes/install.ps1))) -Agent pi
```

Choose `pi`, `opencode`, `hermes`, or `all`. Restart the agent after installation. The installer creates an isolated environment, selects a PyTorch build, downloads the pinned ~0.8 GB model, installs the adapter, and warms the local runtime. No sudo or administrator access is needed.

Prefer reviewing scripts first? Download [install.sh](scripts/install.sh), [install.ps1](scripts/install.ps1), and [install.py](scripts/install.py), inspect them, and run the appropriate one. The Python installer accepts `--source /path/to/this/repo` for development.

Existing Python environment:

```sh
python -m pip install 'auto-local @ git+https://github.com/SSHDotCodes/auto@v0.1.0'
auto install pi
```

For a specific GPU build, use the installer or install the appropriate [PyTorch wheel](https://pytorch.org/get-started/locally/) before this package. See the tested and supported configurations in [compatibility](docs/compatibility.md).

## Integrations

| Agent | Integration | Auto denies | Auto needs review |
|---|---|---|---|
| Pi | `tool_call` extension | Blocks the call | Prompts in UI; blocks in headless mode |
| OpenCode | `tool.execute.before` plugin | Throws a blocking tool error | Blocks with an actionable message |
| Hermes | `pre_tool_call` plugin | Returns the host's block directive | Requests Hermes's native human approval |
| Other agents | JSON CLI / local HTTP API | Host maps the decision | Host must require review |

An Auto approval allows the call to proceed through the host's existing permissions. It does not erase deny rules, disable sandboxes, or automatically answer unrelated permission prompts. Adapters read the actual user request and completed tool history. Missing context, invalid responses, timeouts, or oversized context never result in an automatic approval.

## Runtime

```sh
auto doctor                         # actual backend, model revision, local paths
auto start                          # warm or reuse the local process
auto stop                           # unload the model
auto configure --device cpu         # force CPU
auto configure --device mps         # Apple Silicon
auto configure --device cuda        # NVIDIA CUDA or AMD ROCm
auto configure --max-tokens 32768   # opt into a larger context budget
auto uninstall pi                   # remove adapter; retain downloaded model
```

The default device is `auto`: usable CUDA/ROCm, then Apple MPS, then CPU. GPU initialization and kernel failures fall back to CPU where possible. `auto doctor` reports the device actually used. No FlashAttention compilation is required: the default backend computes exact local/global attention in bounded query blocks. For optional FlashAttention, install this repository with its `flash` extra in the same Python environment, then run `auto configure --attention flash` on a compatible NVIDIA system.

The model supports 65,536 tokens. The runtime defaults to an **8,192-token interactive budget**; increase it up to 65,536 according to memory and latency requirements. Longer inputs request review, **without truncating away instructions or evidence**. Long-context inference on CPU can be very slow. There is no claim that all GPU generations have the same speed or numerical behavior.

The local service starts on demand, binds only `127.0.0.1` on an available port, requires a per-run bearer token, and rejects browser-origin requests. Its private runtime state stays in the user data directory. It logs no prompts or tool arguments. Set `AUTO_HOME` to override the data directory; host proxies are ignored for local requests.

## Measured model results

Same 3,000-example Approve-or-Deny evaluation, full input lengths, threshold 0.5:

| Model | Accuracy | False approve | False deny |
|---|---:|---:|---:|
| Original auto-0.4b | 89.63% | 6.07% | 14.13% |
| auto-1b-bf16 | 96.60% | 4.14% | 2.75% |
| **auto-0.4b-2** | **96.77%** | **3.14%** | **3.31%** |

Independent validation audit: **97.88%** (2,595 examples). The 16k–64k benchmark slice: **94.56%** (239 examples). The 0.17-point lead over auto-1b is five examples, not evidence of universal superiority. These are the published BF16/FlashAttention reference measurements; backend smoke checks do not establish identical full-benchmark accuracy on every device. [Full model evaluation and methodology](https://huggingface.co/ProCreations/auto-0.4b-2).

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

Responses contain `decision` (`approve`, `deny`, or `review`), `p_deny`, and `reason`. Review responses may omit the probability. The CLI only classifies text; it never executes the proposed call. See [architecture](docs/architecture.md) for protocol and integration details.

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
