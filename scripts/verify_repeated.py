"""Check accelerator allocations after five identical requests through Classifier.score."""

import argparse
import gc
import json
import time
from pathlib import Path

import torch

from auto_gate.config import Settings
from auto_gate.model import Classifier
from auto_gate.schema import ScoreRequest

p = argparse.ArgumentParser()
p.add_argument("--device", required=True)
p.add_argument("--model", required=True)
p.add_argument("--output", required=True)
a = p.parse_args()
c = Classifier(Settings(device=a.device, model=a.model, max_tokens=65536))
sync = {"mps": torch.mps.synchronize, "cuda": torch.cuda.synchronize}.get(a.device, lambda: None)
read = {"mps": torch.mps.current_allocated_memory, "cuda": torch.cuda.memory_allocated}.get(
    a.device, lambda: 0
)
request = ScoreRequest(
    user_request="Read README. " + ("Keep all records. " * 1600),
    call={"tool": "read_file", "args": {"path": "README.md"}},
)
rows = []
for i in range(5):
    start = time.perf_counter()
    result = c.score(request)
    sync()
    gc.collect()
    rows.append(
        {
            "seconds": time.perf_counter() - start,
            "tokens": result.tokens,
            "decision": result.decision,
            "bytes_after": read(),
            "p_deny": result.p_deny,
        }
    )
    assert all(getattr(m, "_expanded", None) is None for m in c.model.modules())
assert len({r["decision"] for r in rows}) == 1
assert rows[0]["decision"] in {"approve", "deny"}
assert rows[-1]["bytes_after"] <= rows[0]["bytes_after"] + 2**20
Path(a.output).write_text(json.dumps({"runtime": c.info(), "rows": rows}, indent=2) + "\n")
