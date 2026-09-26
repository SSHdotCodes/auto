# Hardware and operating systems

Auto uses PyTorch and computes ModernBERT's exact attention with memory that grows linearly with context (fused scaled-dot-product attention for global layers, window-only blocks for sliding-window layers, chunked MLP). It does not require a CUDA compiler or a FlashAttention extension. One unpadded request is evaluated at a time. If the optional FlashAttention kernel cannot load, Auto uses its portable attention on the same GPU; other GPU failures fall back to CPU where PyTorch can still run. `auto doctor` reports the actual device, attention path and any fallback.

## Supported installation paths

| System | Backend | Scope |
|---|---|---|
| macOS, Apple Silicon | MPS, FP32 | M-series Macs supported by the installed PyTorch/macOS combination; CPU fallback |
| Windows x86-64 | CPU or NVIDIA CUDA | Python 3.10–3.13; modern supported Windows, current graphics driver |
| Linux x86-64 | CPU, NVIDIA CUDA, AMD ROCm | Modern glibc distribution supported by the selected wheel |
| Linux aarch64 | CPU | Requires an available PyTorch CPU wheel and sufficient RAM; 64-bit OS only |
| Windows with AMD/Intel GPU | CPU | This release does not install DirectML or Windows ROCm |
| Intel macOS | Unavailable natively | Current compatible PyTorch releases no longer ship Intel macOS wheels; use a supported Linux environment |

There is no promise that every historical GPU or operating-system release can run a current accelerator stack. Unsupported GPUs can still use the CPU build on a supported OS. 32-bit operating systems are not supported. Jetson's vendor-specific CUDA distribution is not auto-installed; use CPU or provision a compatible vendor environment yourself.

Allow roughly **3 GB free RAM at minimum for short CPU requests** with auto-0.4b-2 (about 1.5 GB with auto-200m-2) and several GB of disk for Python, dependencies, and the weights (~0.8 GB or ~0.3 GB). GPU distributions can take much more disk space. The default budget is 32,768 tokens. Memory grows linearly with context: a full 65,536-token request adds about 3 GB with auto-0.4b-2 or about 2 GB with auto-200m-2 on CPU or Apple MPS (FP32). Global attention is still quadratic in total work, so long requests are slow on small computers; see the latency table in the [README](../README.md#long-contexts).

## Installer build selection

The automatic installer reads NVIDIA compute capability and driver major version; no device-name substring guessing is used. You can override it:

```sh
curl -fsSL https://auto.ssh.codes/install.sh | sh -s -- --agent opencode --backend cuda-legacy
# or run the reviewed Python installer:
python scripts/install.py --agent hermes --backend cpu
```

| Selection | PyTorch | Distribution | Typical devices / requirements |
|---|---|---|---|
| `cpu` | 2.13.0 | Official CPU wheels | Supported Windows/Linux; arm64 macOS CPU via the unified wheel |
| `mps` | 2.13.0 | PyPI macOS arm64 wheel | Apple M-series; supported macOS |
| `cuda` | 2.13.0 | CUDA 13.0 | Compute capability ≥7.5, driver ≥580; includes Blackwell |
| `cuda12` | 2.8.0 | CUDA 12.8 | Capability ≥7.5, driver ≥570; includes Blackwell |
| `cuda-legacy` | 2.7.1 | CUDA 11.8 | Maxwell/Pascal/Volta and later architectures included in that wheel; capability ≥5 and <10, driver ≥520 |
| `rocm` | 2.13.0 | ROCm 7.1 | Supported Linux AMD devices with working ROCm drivers |

Auto-selection chooses the newest applicable row. A missing accelerator wheel falls back to the CPU wheel in auto mode; an explicitly selected unavailable build fails with an error. It never installs or changes GPU drivers. Older NVIDIA cards use FP32 if native BF16 is unavailable. Older than Maxwell, unsupported drivers, and unrecognized devices use CPU.

AMD support follows the official [ROCm compatibility matrix](https://rocm.docs.amd.com/en/latest/compatibility/compatibility-matrix.html), not a blanket claim for every Radeon card. Confirm your GPU and OS there. Use the official [PyTorch installation selector](https://pytorch.org/get-started/locally/) for custom installations and [Apple MPS guidance](https://developer.apple.com/metal/pytorch/) for OS requirements. Do not use GPU architecture spoofing as a substitute for supported drivers.

## What was physically tested for v0.2.0

The same 24 authored tool probes, plus review for missing and oversized context, for both models:

| Machine | Actual backend | auto-0.4b-2 | auto-200m-2 |
|---|---|---|---|
| Apple M4 Max, macOS 27, PyTorch 2.13.0 | CPU FP32, portable attention | 24/24 | 24/24 |
| Apple M4 Max, macOS 27, PyTorch 2.13.0 | MPS FP32, portable attention | 24/24 | 24/24 |
| RTX PRO 6000 Blackwell, Ubuntu 24.04, PyTorch 2.13.0+cu130 | CUDA BF16, portable attention | 24/24 | 24/24 |
| RTX PRO 6000 Blackwell, same | CUDA BF16, FlashAttention (clean `auto download`, telemetry disabled, offline inference) | 24/24 | 24/24 |
| RTX PRO 6000 Blackwell, same | CUDA BF16, kernel missing or unloadable → portable attention on the GPU | — | 24/24 |

Long contexts (2k–64k tokens, one request each, fresh process): both models on M4 Max CPU and MPS and on the RTX PRO 6000; the RTX PRO 6000 also with every fused attention kernel disabled. The full 3,000-item benchmark through the portable path on the RTX PRO 6000: auto-0.4b-2 2,902 vs 2,903 for its published FlashAttention predictions, auto-200m-2 2,889 vs 2,890. Numbers are in the [README](../README.md#long-contexts).

Raw results are in [verification](verification/). The short probes are smoke checks, not the full benchmark or a latency ranking. Model-call latency excludes adapter/IPC overhead and warm-up. CI separately checks CPU inference with both models on Linux, Windows, and macOS, and compatibility with the older PyTorch runtime. AMD, older NVIDIA GPUs, and earlier Apple Silicon generations have **not** been physically tested for this release. Their paths are implemented against the supported frameworks, and the math-kernel-only measurement covers the case of a GPU without memory-efficient attention; contributions with real hardware reports are welcome.

## Troubleshooting

- Run `auto doctor` for backend and model revision. Its report does not include the local bearer token.
- If a driver or GPU kernel fails, use `auto configure --device cpu`, then `auto start`.
- If the model is not cached, run `auto download` while online. Runtime inference never downloads weights implicitly.
- If a session exceeds the budget, Auto requests review without discarding context. `auto configure --max-tokens 65536` allows the full window; memory stays bounded, but check that response time fits your agent's hook timeout. `auto configure --model auto-200m-2` takes about half as long on CPU (see the README table).
- Hermes's hook has a shorter time budget: Auto gives it up to 20 seconds before requesting native human approval. Warm the model with `auto start` before using Hermes on a slow CPU.
- Optional FlashAttention is for supported NVIDIA systems: install this repository with the `flash` extra, then `auto configure --attention flash` and `auto download`, which stores the pinned kernel in Auto's data directory for offline use. If the kernel is missing or cannot load, `auto doctor` shows the reason and Auto uses the portable attention on the same GPU. The portable backend already uses PyTorch's fused attention kernels on CUDA and matched FlashAttention on the full benchmark (below), so FlashAttention mainly saves time.
- On Windows, use the full CLI path printed by the installer if `auto` is not on PATH. Agent adapters store an absolute Python path and do not depend on PATH.
