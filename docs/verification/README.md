# Verification records (v0.2.0)

Raw results behind the claims in the [README](../../README.md#long-contexts) and [compatibility](../compatibility.md).

## Tool probes (24 authored probes, plus review for missing and oversized context)

| File | Machine | Model | Runtime path |
|---|---|---|---|
| [apple-m4-max-cpu.json](apple-m4-max-cpu.json) | Apple M4 Max | auto-0.4b-2 | CPU FP32, portable attention |
| [apple-m4-max-mps.json](apple-m4-max-mps.json) | Apple M4 Max | auto-0.4b-2 | MPS FP32, portable attention |
| [apple-m4-max-cpu-auto-200m-2.json](apple-m4-max-cpu-auto-200m-2.json) | Apple M4 Max | auto-200m-2 | CPU FP32, portable attention |
| [apple-m4-max-mps-auto-200m-2.json](apple-m4-max-mps-auto-200m-2.json) | Apple M4 Max | auto-200m-2 | MPS FP32, portable attention |
| [rtx-pro-6000-cuda.json](rtx-pro-6000-cuda.json) | RTX PRO 6000 | both | CUDA BF16: FlashAttention after a clean `auto download`, portable attention, and the fallback when the kernel is missing or cannot load |

Produced by [scripts/verify_model.py](../../scripts/verify_model.py) and [harness/flash_e2e.py](harness/flash_e2e.py).

## Long contexts

| File | What |
|---|---|
| [long-context/apple-m4-max-*.json](long-context/) | Peak memory and time of one request at 2k–64k tokens, each length in a fresh process, [scripts/measure_long_context.py](../../scripts/measure_long_context.py); `apple-m4-max-v0.1.0.json` is Auto v0.1.0 measured the same way |
| [long-context/rtx-pro-6000-cuda-auto-0.4b-2.json](long-context/rtx-pro-6000-cuda-auto-0.4b-2.json), [...-auto-200m-2.json](long-context/rtx-pro-6000-cuda-auto-200m-2.json) | The same measurement on CUDA (BF16) |
| [long-context/rtx-pro-6000-cuda-math-only.json](long-context/rtx-pro-6000-cuda-math-only.json) | CUDA with every fused attention kernel disabled (PyTorch's plain math kernel only), the worst case for a backend without memory-efficient attention; peak memory and p(deny) against the fused run |
| [long-context/rtx-pro-6000-cuda-benchmark.json](long-context/rtx-pro-6000-cuda-benchmark.json) | The full 3,000-item Approve-or-Deny benchmark (revision `a38b6259`, full input lengths up to 64k tokens) through the portable attention in BF16, compared item by item with each model's published FlashAttention predictions; plus peak CUDA memory of stock Transformers SDPA at 8k and 16k tokens for contrast |

The CUDA harness scripts in [harness/](harness/) ran inside the temporary training job that produced auto-200m-2; their paths refer to that job's layout (tokenized benchmark, reference weights and predictions).
