# Hardware and operating systems

Auto uses PyTorch and exact, chunked scaled-dot-product attention. It does not require a CUDA compiler or a FlashAttention extension. One unpadded request is evaluated at a time. GPU failures fall back to CPU where PyTorch can still run; `auto doctor` reports the actual device and any fallback.

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

Allow roughly **3 GB free RAM at minimum for short CPU requests** and several GB of disk for Python, dependencies, and the ~0.8 GB weights. GPU distributions can take much more disk space. The default interactive limit is 8,192 tokens; memory and latency grow with context, and global attention remains quadratic in total work. Opting into 65,536 tokens does not make that workload fast on a small computer.

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

## What was physically tested for v0.1.0

The same 24 authored tool probes, plus review for missing and oversized context:

| Machine | Actual backend | Result |
|---|---|---|
| Apple M4 Max, macOS 27, PyTorch 2.13.0 | CPU FP32 | 24/24 |
| Apple M4 Max, macOS 27, PyTorch 2.13.0 | MPS FP32 | 24/24 |
| RTX PRO 6000 Blackwell, Ubuntu, PyTorch 2.13.0+cu130 | CUDA BF16 | 24/24 |

Raw results are in [verification](verification/). These short illustrative probes are smoke checks, not the full 3,000-example benchmark or a latency ranking. Model-call latency excludes adapter/IPC overhead and warm-up. CI separately checks CPU inference on Linux, Windows, and macOS, and compatibility with the older PyTorch runtime. AMD, older NVIDIA GPUs, and earlier Apple Silicon generations have **not** been physically tested for this release. Their paths are implemented against the supported frameworks; contributions with real hardware reports are welcome.

## Troubleshooting

- Run `auto doctor` for backend and model revision. Its report does not include the local bearer token.
- If a driver or GPU kernel fails, use `auto configure --device cpu`, then `auto start`.
- If the model is not cached, run `auto download` while online. Runtime inference never downloads weights implicitly.
- If a session exceeds the budget, Auto requests review without discarding context. Increase `auto configure --max-tokens 16384` only if memory and response time permit.
- Hermes's hook has a shorter time budget: Auto gives it up to 20 seconds before requesting native human approval. Warm the model with `auto start` before using Hermes on a slow CPU.
- Optional FlashAttention is for supported NVIDIA systems: install this repository with the `flash` extra, then `auto configure --attention flash`. The default chunked backend is the portable path.
- On Windows, use the full CLI path printed by the installer if `auto` is not on PATH. Agent adapters store an absolute Python path and do not depend on PATH.
