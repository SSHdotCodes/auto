# Auto v0.3.0 verification

Measured September 27, 2026 using Transformers 5.16.1 and the installer’s PyTorch 2.13.0 build. macOS 27 / Apple M4 Max uses FP32 on CPU and MPS; Ubuntu 24.04 / RTX PRO 6000 Blackwell uses CUDA BF16. The runtime’s default portable attention was used, with no optional FlashAttention extension.

## Long-context latency and memory

One synthetic unpadded request per fresh process, following the runtime’s tiny startup warm-up. Times are single observations, not sustained throughput. Memory is peak **above the loaded model**, in GiB (2³⁰ bytes). CPU uses RSS polling; MPS uses Metal driver allocation polling; CUDA uses its allocator high-water mark. Allocation sampling is approximate. Full settings and loaded baselines are in each JSON file.

| Model | Device | 2k time | 16k time / extra GiB | 32k time / extra GiB | 64k time / extra GiB |
|---|---|---:|---:|---:|---:|
| [int4](int4-cpu-memory.json) | CPU | 0.393 s | 5.93 s / +0.66 | 19.54 s / +1.08 | 71.60 s / +2.03 |
| [int8](int8-cpu-memory.json) | CPU | 0.365 s | 5.93 s / +0.66 | 20.05 s / +1.08 | 74.21 s / +1.98 |
| [int4](int4-mps-memory.json) | MPS | 0.121 s | 1.50 s / +0.00 | 6.49 s / +1.00 | 24.70 s / +1.66 |
| [int8](int8-mps-memory.json) | MPS | 0.135 s | 1.53 s / +0.00 | 5.39 s / +1.00 | 24.66 s / +1.66 |
| [int4](auto-200m-2-int4-memory.json) | CUDA | 0.011 s | 0.06 s / +0.34 | 0.17 s / +0.67 | 0.49 s / +1.34 |
| [int8](auto-200m-2-int8-memory.json) | CUDA | 0.009 s | 0.06 s / +0.34 | 0.17 s / +0.67 | 0.50 s / +1.34 |

Metal reserves memory in coarse heaps. A reported +0.00 GiB means the request fit in existing driver allocations, not that it needed no activation memory. The quantized MPS loaded driver baseline is roughly 1 GiB despite the smaller live weight tensors. CPU process RSS also includes Python/PyTorch, and CUDA RSS is separate from VRAM.

The model weights stay packed, but activations stay floating point. Quantization does not make total inference memory four/eight times smaller. Long-context memory remains linear; global-attention work remains quadratic. The default limit remains 32,768 tokens and over-budget requests ask for review without truncation.

## Worst-case attention fallback

With every fused SDPA kernel disabled, including local attention, both quants complete 8k–64k CUDA requests. This exercises the algorithm used when a device has no memory-efficient fused attention. It is not a hardware test of ROCm or older GPUs.

| Model | 8k extra GiB | 16k extra GiB | 32k extra GiB | 64k extra GiB |
|---|---:|---:|---:|---:|
| [int4](auto-200m-2-int4-math-only.json) | 2.41 | 2.55 | 2.83 | 3.41 |
| [int8](auto-200m-2-int8-math-only.json) | 2.41 | 2.55 | 2.83 | 3.41 |

Unit tests compare against independently reconstructed dense weights, exercise block boundaries, and instrument tensor sizes at 4k and 8k with fused kernels disabled. They also verify CPU fallback and cleanup when an MLP raises an error.

## Full pinned benchmark

All 3,000 rows from `ProCreations/approve-or-deny` at revision `a38b625913dd46ca9702063f1597c979e6ace34e`, original order, full token lengths, deny at P(deny) ≥ 0.5. Model revisions are pinned in `config.py`; exported labels and source rows are checked against the published prediction files.

| Model | Runtime correct | False approve | False deny | Published reference correct | Decisions differing | 16k–64k correct / 239 |
|---|---:|---:|---:|---:|---:|---:|
| [int4](auto-200m-2-int4-benchmark.json) | 2885/3,000 | 60 | 55 | 2884/3,000 | 1 | 227 |
| [int8](auto-200m-2-int8-benchmark.json) | 2889/3,000 | 53 | 58 | 2890/3,000 | 1 | 225 |

Raw logits, labels and token lengths accompany each JSON. These small backend-dependent differences are disclosed, not evidence of an accuracy improvement. The published quant model cards describe their training and evaluation limitations. No full-benchmark CPU/MPS or ROCm parity claim is made.

## Short calls, repeated requests, and integration

| Model | CPU median | MPS median | CUDA median | Authored probes |
|---|---:|---:|---:|---|
| int4 | 61.4 ms | 29.3 ms | 8.5 ms | 24/24 on each device |
| int8 | 36.7 ms | 22.0 ms | 6.0 ms | 24/24 on each device |

These authored probes are smoke tests, not an independent quality benchmark. Timings exclude adapter and HTTP overhead. Missing and oversized context both return review. Five identical 6,452-token requests through `Classifier.score` leave stable live accelerator allocations and no expanded-weight cache on MPS/CUDA (`*-repeated.json`).

Linux, Windows and macOS CI run real weights for all four variants, offline daemons and installed adapters; Linux also covers PyTorch 2.7.1/Python 3.10. Integration tests switch a live int4 daemon to int8 and verify the next request uses the new model. Python tests, adapter/host contracts, lint/build/package checks and the desktop/mobile website suite are release gates.

Optional pinned FlashAttention was checked separately at 65,536 tokens (`flash.json`); the recorded runtime field distinguishes actual FlashAttention from portable fallback. AMD ROCm, older NVIDIA hardware and other Apple generations are supported through the portable framework path but were not physically measured.

## Reproduce

Use an isolated `AUTO_HOME`, install `.[dev]`, and download the desired pinned model. For example:

```sh
auto download --model auto-200m-2-int4
python scripts/verify_model.py --device cpu --model auto-200m-2-int4 --output smoke.json
python scripts/measure_long_context.py --device mps --model auto-200m-2-int4 --output memory.json
python scripts/verify_repeated.py --device mps --model auto-200m-2-int4 --output repeated.json
# Full CUDA benchmark + forced-math sweep also requires datasets and numpy:
python scripts/verify_quantized.py --model auto-200m-2-int4 --output results
```

Repeat with `auto-200m-2-int8`. The [PyTorch 2.14 observations](pytorch214/) include another full CUDA benchmark and CPU/MPS/CUDA memory sweeps, plus the unquantized 200M model on CPU/MPS. They are kept separately from the default-installer measurements; single-run latency varies with hardware load and framework version.
