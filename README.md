# Auto

[![CI](https://github.com/SSHDotCodes/auto/actions/workflows/ci.yml/badge.svg)](https://github.com/SSHDotCodes/auto/actions/workflows/ci.yml)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-22543d.svg)](LICENSE)

![Auto — local permission intelligence](site/assets/mountain.webp)

**Local permission intelligence for coding agents.**

[Website](https://auto.ssh.codes) · Models: [auto-0.4b-2](https://huggingface.co/ProCreations/auto-0.4b-2), [auto-200m-2](https://huggingface.co/ProCreations/auto-200m-2) · [Compatibility](docs/compatibility.md) · [Agent integrations](docs/agents.md)

Auto checks a proposed tool call against the user's request and the agent's tool history. A ModernBERT classifier runs on your computer; prompts stay local. Model files download during installation, then inference works offline. Two models are available: **auto-0.4b-2** (395.8M parameters, the default) and the smaller, faster **auto-200m-2** (149.6M). Both read up to 65,536 tokens.

## Install

**macOS / Linux** (downloads uv and Python 3.13 if needed):

```sh
curl -fsSL https://auto.ssh.codes/install.sh | sh -s -- --agent pi
```

**Windows PowerShell**:

```powershell
& ([scriptblock]::Create((irm https://auto.ssh.codes/install.ps1))) -Agent pi
```

Choose `pi`, `opencode`, `hermes`, or `all`. Restart the agent after installation. The installer creates an isolated environment, selects a PyTorch build, downloads the pinned model (~0.8 GB), installs the adapter, and warms the local runtime. No sudo or administrator access is needed. For the smaller model add `--model auto-200m-2` (PowerShell: `-Model auto-200m-2`; ~0.3 GB).

Prefer reviewing scripts first? Download [install.sh](scripts/install.sh), [install.ps1](scripts/install.ps1), and [install.py](scripts/install.py), inspect them, and run the appropriate one. The Python installer accepts `--source /path/to/this/repo` for development.

Existing Python environment:

```sh
python -m pip install 'auto-local @ git+https://github.com/SSHDotCodes/auto@v0.2.0'
auto install pi                     # or: auto install pi --model auto-200m-2
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
auto configure --model auto-200m-2  # smaller, faster model (auto download fetches it)
auto configure --max-tokens 65536   # the model's full context
auto uninstall pi                   # remove adapter; retain downloaded model
```

The default device is `auto`: usable CUDA/ROCm, then Apple MPS, then CPU. If the optional FlashAttention kernel cannot load, Auto uses its portable attention on the same GPU; other GPU initialization failures fall back to CPU where possible. `auto doctor` reports the device and attention actually used. No FlashAttention compilation is required. For optional FlashAttention, install this repository with its `flash` extra in the same Python environment, run `auto configure --attention flash` on a compatible NVIDIA system, then `auto download` to fetch the pinned kernel.

### Long contexts

Both models read 65,536 tokens. The runtime scores up to **32,768 tokens by default** (`auto configure --max-tokens 65536` for the full window); longer inputs request review, **without truncating away instructions or evidence**.

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
