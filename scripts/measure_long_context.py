"""Peak memory and latency of one Auto forward pass versus context length, each length in a fresh process.

Uses the installed runtime (the same attention path as `auto serve`) on synthetic token ids; content does not
affect memory or time. Nothing is executed. Example:

    python scripts/measure_long_context.py --device mps --model auto-200m-2 --output result.json
"""

import argparse
import json
import os
import platform
import subprocess
import sys
import threading
import time

LENGTHS = [2048, 8192, 16384, 32768, 65536]


def child(device, model, length):
    import psutil
    import torch

    from auto_gate.config import Settings
    from auto_gate.model import Classifier

    classifier = Classifier(Settings(device=device, model=model, max_tokens=65536))
    if classifier.device != device:
        raise SystemExit(f"requested {device}, runtime chose {classifier.device}")
    sync = {"cuda": torch.cuda.synchronize, "mps": torch.mps.synchronize}.get(device, lambda: None)
    sync()
    # MPS allocations do not appear in RSS; poll the Metal driver's view instead.
    read = {
        "cuda": torch.cuda.memory_allocated,
        "mps": torch.mps.driver_allocated_memory,
    }.get(device)
    process = psutil.Process()
    loaded_rss, loaded = process.memory_info().rss, read() if read else 0
    peak, peak_rss, stop = [loaded], [loaded_rss], threading.Event()
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()

    def poll():
        while not stop.is_set():
            peak_rss[0] = max(peak_rss[0], process.memory_info().rss)
            if read and device != "cuda":
                peak[0] = max(peak[0], read())
            time.sleep(0.002)

    threading.Thread(target=poll, daemon=True).start()
    ids = torch.randint(1000, 50000, (1, length), generator=torch.Generator().manual_seed(0))
    ids[0, 0], ids[0, -1] = classifier.tokenizer.cls_token_id, classifier.tokenizer.sep_token_id
    started = time.perf_counter()
    with torch.inference_mode():
        logits = classifier._forward(ids.to(classifier.device))
        sync()
    seconds = time.perf_counter() - started
    stop.set()
    if device == "cuda":
        peak[0] = torch.cuda.max_memory_allocated()
    row = {
        "device": device,
        "model": model,
        "tokens": length,
        "seconds": round(seconds, 4),
        "runtime": classifier.info(),
        "loaded_rss_gb": round(loaded_rss / 2**30, 3),
        "loaded_accelerator_gb": round(loaded / 2**30, 3),
        "attention": classifier.attention,
        "finite": bool(torch.isfinite(logits).all()),
    }
    if read:
        row["accelerator_gb_over_loaded"] = round((peak[0] - loaded) / 2**30, 2)
    row["rss_gb_over_loaded"] = round((peak_rss[0] - loaded_rss) / 2**30, 2)
    print(json.dumps(row), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--device", required=True, choices=["cpu", "mps", "cuda"])
    parser.add_argument("--model", default="auto-0.4b-2")
    parser.add_argument("--lengths", type=int, nargs="*", default=LENGTHS)
    parser.add_argument("--output", required=True)
    parser.add_argument("--child", type=int, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.child:
        return child(args.device, args.model, args.child)
    import torch

    rows = []
    for length in args.lengths:
        command = [sys.executable, __file__, "--device", args.device, "--model", args.model, "--output", "-"]
        result = subprocess.run([*command, "--child", str(length)], capture_output=True, text=True)
        if result.returncode:
            rows.append({"tokens": length, "error": result.stderr.strip().splitlines()[-1:]})
            break
        rows.append(json.loads(result.stdout.strip().splitlines()[-1]))
        print(json.dumps(rows[-1]), flush=True)
    report = {
        "platform": platform.platform(),
        "processor": platform.processor() or platform.machine(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "budget_mb_override": os.environ.get("AUTO_ATTENTION_BUDGET_MB"),
        "method": "One forward pass per fresh process on synthetic token ids; memory is the peak above the loaded model.",
        "results": rows,
    }
    with open(args.output, "w", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")
    if len(rows) != len(args.lengths) or any(row.get("finite") is not True for row in rows):
        raise SystemExit("Long-context verification failed; inspect the saved report")


if __name__ == "__main__":
    main()
