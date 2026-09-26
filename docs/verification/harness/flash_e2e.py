"""End-to-end check of Auto's optional FlashAttention path on CUDA and its fallbacks (24 probes per configuration).

Run with the models and the pinned kernel already fetched by `auto download`; inference itself is offline.
usage: flash_e2e.py OUT_JSON
"""

import json
import os
import sys

sys.path.insert(0, "/job/work/auto-repo/scripts")
from verify_model import verify

import auto_gate.model as runtime
from auto_gate.config import FLASH, Settings
from auto_gate.model import Classifier
from auto_gate.schema import ScoreRequest

probes = json.load(open("/job/work/auto-repo/tests/fixtures/probes.json"))


def run(model, **kw):
    c = Classifier(Settings(device="cuda", model=model, attention="flash"))
    rows = [
        c.score(ScoreRequest(**{k: p[k] for k in ("user_request", "call", "history")})).model_dump()
        for p in probes
    ]
    info = c.info()
    del c
    return info, rows


# Kernel present but unloadable (fresh process, since a loaded kernel stays registered): must fall back to chunked
# attention on the same GPU, not to CPU.
if len(sys.argv) > 2 and sys.argv[2] == "broken":
    os.environ["LOCAL_KERNELS"] = FLASH.split("@")[0] + "=/nonexistent"
    info, rows = run("auto-200m-2")
    assert info["attention"] == "chunked" and info["device"] == "cuda" and info["fallback"], info
    out = {
        "kernel_broken": {
            "runtime": info,
            "passed": sum(r["decision"] == p["expected"] for r, p in zip(rows, probes)),
        }
    }
    print("kernel_broken", json.dumps(out["kernel_broken"]), flush=True)
    json.dump(out, open(sys.argv[1], "w"), indent=1, default=str)
    sys.exit()

out = {}
for model in ["auto-0.4b-2", "auto-200m-2"]:
    chunked = verify("cuda", model=model)
    base = {r["id"]: r["p_deny"] for r in chunked["results"]}
    info, rows = run(model)
    assert info["attention"] == "flash" and info["device"] == "cuda" and info["fallback"] is None, info
    flash = {
        "runtime": info,
        "passed": sum(r["decision"] == p["expected"] for r, p in zip(rows, probes)),
        "total": len(rows),
        "max_abs_p_deny_vs_chunked": max(abs(r["p_deny"] - base[p["id"]]) for r, p in zip(rows, probes)),
    }
    out[model] = {"chunked": {k: chunked[k] for k in ("passed", "total")}, "flash": flash}
    print(model, json.dumps(out[model]), flush=True)

# Kernel never downloaded: chunked attention on the same GPU, with the reason reported.
real = runtime.flash_kernel_path
runtime.flash_kernel_path = lambda: real().parent / "missing"
info, rows = run("auto-200m-2")
runtime.flash_kernel_path = real
assert info["attention"] == "chunked" and info["device"] == "cuda" and "not downloaded" in info["fallback"], (
    info
)
out["kernel_missing"] = {
    "runtime": info,
    "passed": sum(r["decision"] == p["expected"] for r, p in zip(rows, probes)),
}
print("kernel_missing", json.dumps(out["kernel_missing"]), flush=True)

json.dump(out, open(sys.argv[1], "w"), indent=1, default=str)
