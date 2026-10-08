"""CUDA verification for Auto v0.2.0's portable attention (run inside the auto-200m-2 temporary job after training).

1. Full 3,000-item Approve-or-Deny benchmark, one unpadded request at a time, through auto_gate's chunked attention
   (BF16, the runtime's CUDA configuration) vs each model's published FlashAttention reference logits.
2. Peak CUDA memory vs context length for: chunked (fused SDPA), chunked with fused kernels disabled (math fallback),
   and stock Transformers SDPA for contrast (only where it fits).
usage: verify_longctx.py OUT_JSON MODEL_NAME=MODEL_DIR=REFERENCE_NPZ [...]
"""

import gc
import json
import sys
import time

import numpy as np
import torch
from datasets import load_from_disk
from scipy.special import softmax
from transformers import AutoModelForSequenceClassification
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

from auto_gate import attention
from auto_gate.attention import bound_mlp_memory, chunked_attention, forward_classifier

BENCH = "/job/work/data/benchmark"
ALL_ATTENTION_FUNCTIONS.register("auto_chunked", chunked_attention)


def load(path, impl):
    model = (
        AutoModelForSequenceClassification.from_pretrained(
            path, dtype=torch.bfloat16, attn_implementation=impl
        )
        .cuda()
        .eval()
    )
    return bound_mlp_memory(model) if impl == "auto_chunked" else model


@torch.inference_mode()
def benchmark(path, reference):
    bench = load_from_disk(BENCH)
    labels = np.array(bench["labels"])
    model = load(path, "auto_chunked")
    logits = np.zeros((len(bench), 2), np.float32)
    t = time.time()
    for i in range(len(bench)):
        ids = torch.tensor(bench[i]["input_ids"], device="cuda")[None]
        logits[i] = forward_classifier(model, ids).float().cpu().numpy()[0]
    seconds = time.time() - t
    ref = np.load(reference)["logits"].astype(np.float64)
    p, pr = softmax(logits.astype(np.float64), 1)[:, 1], softmax(ref, 1)[:, 1]
    lens = np.array(bench["length"])
    long = lens >= 16384
    out = {
        "n": len(bench),
        "seconds": seconds,
        "chunked_correct": int(((p >= 0.5) == labels).sum()),
        "reference_correct": int(((pr >= 0.5) == labels).sum()),
        "decision_agreement": float(((p >= 0.5) == (pr >= 0.5)).mean()),
        "decisions_differing": int(((p >= 0.5) != (pr >= 0.5)).sum()),
        "max_abs_p_deny_difference": float(np.abs(p - pr).max()),
        "median_abs_logit_difference": float(np.median(np.abs(logits - ref))),
        "long_16k_64k": {
            "n": int(long.sum()),
            "chunked_correct": int(((p >= 0.5) == labels)[long].sum()),
            "reference_correct": int(((pr >= 0.5) == labels)[long].sum()),
        },
    }
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return out, logits


def peak(path, impl, n, fused=True):
    saved = dict(attention._FUSED)
    if not fused:
        attention._FUSED.clear()
    try:
        model = load(path, impl)
        torch.cuda.synchronize()
        gc.collect()
        torch.cuda.empty_cache()
        base = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        ids = torch.randint(1000, 50000, (1, n), device="cuda")
        ids[0, 0] = 50281
        ids[0, -1] = 50282
        t = time.time()
        with torch.inference_mode():
            logits = forward_classifier(model, ids) if impl == "auto_chunked" else model(input_ids=ids).logits
        torch.cuda.synchronize()
        row = {
            "impl": impl if fused else impl + " (fused kernels disabled)",
            "tokens": n,
            "seconds": round(time.time() - t, 3),
            "peak_over_weights_gb": round((torch.cuda.max_memory_allocated() - base) / 2**30, 3),
            "finite": bool(torch.isfinite(logits).all()),
        }
    except torch.OutOfMemoryError:
        row = {"impl": impl if fused else impl + " (fused kernels disabled)", "tokens": n, "oom": True}
    finally:
        attention._FUSED.clear()
        attention._FUSED.update(saved)
        attention._UNAVAILABLE.clear()
        model = None
        gc.collect()
        torch.cuda.empty_cache()
    print(json.dumps(row), flush=True)
    return row


def main():
    out = {"device": torch.cuda.get_device_name(), "torch": torch.__version__, "models": {}}
    for spec in sys.argv[2:]:
        name, path, reference = spec.split("=")
        result, logits = benchmark(path, reference)
        print(name, json.dumps(result), flush=True)
        memory = []
        for n in (8192, 16384, 32768, 65536):
            memory.append(peak(path, "auto_chunked", n))
            memory.append(peak(path, "auto_chunked", n, fused=False))
        for n in (8192, 16384):
            memory.append(peak(path, "sdpa", n))
        out["models"][name] = {"benchmark": result, "memory": memory}
        np.savez(sys.argv[1].replace(".json", f"-{name}-chunked-logits.npz"), logits=logits)
    json.dump(out, open(sys.argv[1], "w"), indent=2)


if __name__ == "__main__":
    main()
