# Changelog

## 0.2.0

- **auto-200m-2**, a 149.6M-parameter ModernBERT-base model with the same 65,536-token context (96.33% on the
  3,000-item benchmark vs 96.77% for the pinned auto-0.4b-2), selectable with `--model auto-200m-2` in the CLI and
  all installers. It takes about half as long on CPU and needs about 1 GB of RAM once loaded instead of 1.9 GB.
  auto-0.4b-2 stays the default.
- **Long contexts no longer blow up memory where FlashAttention is unavailable.** Global layers use PyTorch's fused
  attention (or bounded query blocks where no fused kernel exists), sliding-window layers attend each block only to
  its window, and the per-token MLP runs in sequence chunks. No N×N score matrix or mask is allocated on CPU, Apple
  MPS, CUDA without FlashAttention, or ROCm. On an M4 Max, a 65,536-token auto-0.4b-2 request now needs +3.1 GB
  on MPS instead of +12.4 GB and takes 42 s instead of 366 s; on CPU, +2.9 GB and 142 s instead of +4.7 GB and
  810 s. On an RTX PRO 6000 it needs +1.8 GB; with every fused attention kernel disabled, at most +3.8 GB. The full
  benchmark through this path on CUDA came within one item of the published FlashAttention results for both
  models. Short requests are unchanged in output and slightly faster.
- Default context budget raised from 8,192 to 32,768 tokens; v0.1.0 settings that still had the old 8,192 default
  move to the new one, other saved budgets are kept.
- Optional FlashAttention works again on current installs: `kernels` pinned to 0.16.x (the range Transformers
  5.16.1 accepts); `auto download` stores the pinned kernel in Auto's data directory, also when
  `HF_HUB_DISABLE_TELEMETRY` is set, and the runtime loads that copy offline (kernels 0.16 cannot load a cached hub
  kernel offline by itself). If the kernel is missing or fails to load, Auto now uses its portable attention on the
  same GPU instead of falling back to CPU, and `auto doctor` reports why.
- Startup runs a tiny warm-up pass so GPU and kernel failures are caught before the first request.
- v0.1.0 configuration files load unchanged.

## 0.1.0

- Initial local runtime with pinned Auto-0.4b-2 weights and no inference network requirement.
- Pi, OpenCode, and Hermes integrations preserving user context and native permission boundaries.
- Isolated shell, PowerShell, and Python installers with automatic or explicit CPU/GPU builds.
- Portable exact attention for CPU, Apple MPS, NVIDIA CUDA, and supported AMD ROCm environments.
- Cross-platform CI, real-model smoke checks, and a responsive website with sourced evaluation charts.
