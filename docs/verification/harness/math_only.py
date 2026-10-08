"""Peak CUDA memory of Auto's chunked attention when only PyTorch's plain math SDPA kernel is allowed (no fused
kernel of any kind, the worst case for a backend without memory-efficient attention), plus exactness vs the fused run.
usage: math_only.py OUT_JSON NAME=MODEL_DIR [...]
"""

import gc
import json
import sys
import time

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

from auto_gate import attention
from auto_gate.attention import forward_classifier

sys.path.insert(0, "/job/work/auto-repo-verify")
from verify_longctx import load


def run(model, ids, math):
    saved = dict(attention._FUSED)
    if math:
        attention._FUSED.clear()
    try:
        torch.cuda.synchronize()
        gc.collect()
        torch.cuda.empty_cache()
        base = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        t = time.time()
        with torch.inference_mode(), sdpa_kernel([SDPBackend.MATH]) if math else torch.enable_grad():
            logits = forward_classifier(model, ids).float().cpu()
        torch.cuda.synchronize()
        return logits, round(time.time() - t, 3), round((torch.cuda.max_memory_allocated() - base) / 2**30, 3)
    finally:
        attention._FUSED.clear()
        attention._FUSED.update(saved)
        attention._UNAVAILABLE.clear()


out = {
    "device": torch.cuda.get_device_name(),
    "torch": torch.__version__,
    "score_budget_mb": 1024,
    "models": {},
}
for spec in sys.argv[2:]:
    name, path = spec.split("=")
    model = load(path, "auto_chunked")
    rows = []
    for n in (8192, 16384, 32768, 65536):
        ids = torch.randint(
            1000, 50000, (1, n), device="cuda", generator=torch.Generator("cuda").manual_seed(n)
        )
        ids[0, 0] = 50281
        ids[0, -1] = 50282
        fused, fs, fp = run(model, ids, False)
        math, ms, mp = run(model, ids, True)
        pf, pm = fused.softmax(-1)[0, 1].item(), math.softmax(-1)[0, 1].item()
        rows.append(
            {
                "tokens": n,
                "fused_seconds": fs,
                "fused_peak_gb": fp,
                "math_seconds": ms,
                "math_peak_gb": mp,
                "finite": bool(torch.isfinite(math).all()),
                "p_deny_fused": pf,
                "p_deny_math": pm,
            }
        )
        print(name, json.dumps(rows[-1]), flush=True)
    out["models"][name] = rows
    del model
    gc.collect()
    torch.cuda.empty_cache()
json.dump(out, open(sys.argv[1], "w"), indent=2)
