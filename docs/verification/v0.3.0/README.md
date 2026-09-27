# Auto v0.3.0 verification

The release adds native auto-quant-v1 int4/int8 loading with the existing bounded-memory attention path.
This directory contains real runtime measurements. The completed release report will include fresh-process
2k–64k memory/latency sweeps, authored smoke probes, full pinned CUDA benchmark results and a forced-math
attention sweep. CPU CI additionally covers Linux, Windows, macOS and PyTorch 2.7.1.

The checkpoint weights stay packed. This is weight-only quantization, so activation memory is floating point
and grows with context. No cross-device equivalence or GPU speed is inferred from CPU CI alone.
