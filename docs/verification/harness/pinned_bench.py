"""Full-benchmark check of the portable attention on the auto-0.4b-2 weights Auto actually pins (456e153d)."""

import json
import sys

import numpy as np
from huggingface_hub import snapshot_download

sys.path.insert(0, "/job/work/auto-repo-verify")
from verify_longctx import benchmark

REV = "456e153dad2b12db14babb1dbdc8f8ece607e80a"
path = snapshot_download(
    "ProCreations/auto-0.4b-2",
    revision=REV,
    cache_dir="/job/work/verify/hub3",
    allow_patterns=[
        "config.json",
        "model.safetensors",
        "tokenizer.json",
        "tokenizer_config.json",
        "benchmark_predictions.npz",
    ],
)
ref = np.load(f"{path}/benchmark_predictions.npz")
print("keys", {k: ref[k].shape for k in ref.files}, flush=True)
if "source_row" in ref.files:
    assert list(ref["source_row"]) == list(range(3000))
result, logits = benchmark(path, f"{path}/benchmark_predictions.npz")
result["revision"] = REV
print("auto-0.4b-2@456e153d", json.dumps(result), flush=True)
json.dump(result, open(sys.argv[1], "w"), indent=2)
