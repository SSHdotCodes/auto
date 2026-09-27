"""Pinned full-benchmark verification and forced-math memory checks for quantized Auto.

Run on CUDA after downloading both models. Requires datasets and numpy in addition
to the dev dependencies. Writes JSON metrics and raw logits, never dataset text.
"""

import argparse
import gc
import json
import time
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from huggingface_hub import hf_hub_download
from torch.nn.attention import SDPBackend, sdpa_kernel

from auto_gate import attention
from auto_gate.config import Settings
from auto_gate.model import Classifier, download

DATASET = "ProCreations/approve-or-deny"
REVISION = "a38b625913dd46ca9702063f1597c979e6ace34e"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, choices=["auto-200m-2-int4", "auto-200m-2-int8"])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    settings = Settings(model=args.model, device="cuda", max_tokens=65536)
    download(settings)
    ref_path = hf_hub_download(settings.model_id, "benchmark_predictions.npz", revision=settings.revision)
    bench = load_dataset(DATASET, revision=REVISION, split="test")
    classifier = Classifier(settings)
    assert classifier.device == "cuda" and classifier.fallback is None
    print("dataset columns", bench.column_names, flush=True)
    # Each dataset row contains the exact serialized model input.
    texts = bench["text"]
    labels = np.asarray([0 if label == "approve" else 1 for label in bench["label"]])
    logits, lengths = [], []
    started = time.perf_counter()
    with torch.inference_mode():
        for i, text in enumerate(texts):
            ids = classifier.tokenizer(text, return_tensors="pt", truncation=False)["input_ids"]
            assert ids.shape[-1] <= 65536
            lengths.append(ids.shape[-1])
            logits.append(classifier._forward(ids.cuda()).float().cpu().numpy()[0])
            if (i + 1) % 250 == 0:
                print(args.model, i + 1, round(time.perf_counter() - started, 2), flush=True)
    values = np.asarray(logits)
    reference = np.load(ref_path)
    assert np.array_equal(labels, reference["labels"])
    assert np.array_equal(reference["source_row"], np.arange(len(labels)))
    predictions, ref_predictions = values.argmax(-1), reference["logits"].argmax(-1)
    long = np.asarray(lengths) >= 16384
    report = {
        "runtime": classifier.info(),
        "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name(),
        "dataset": DATASET,
        "dataset_revision": REVISION,
        "n": len(labels),
        "seconds": time.perf_counter() - started,
        "correct": int((predictions == labels).sum()),
        "false_approve": int(((predictions == 0) & (labels == 1)).sum()),
        "false_deny": int(((predictions == 1) & (labels == 0)).sum()),
        "reference_correct": int((ref_predictions == labels).sum()),
        "decisions_differing": int((predictions != ref_predictions).sum()),
        "long_context": {"n": int(long.sum()), "correct": int((predictions[long] == labels[long]).sum())},
    }
    np.savez(args.output / f"{args.model}-logits.npz", logits=values, labels=labels, lengths=lengths)
    (args.output / f"{args.model}-benchmark.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)
    rows = []
    # Exercise the actual runtime with all fused kernels unavailable, including local attention.
    attention._FUSED.clear()
    for length in (8192, 16384, 32768, 65536):
        gc.collect()
        torch.cuda.empty_cache()
        base = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        ids = torch.randint(1000, 50000, (1, length), generator=torch.Generator().manual_seed(0)).cuda()
        started = time.perf_counter()
        with torch.inference_mode(), sdpa_kernel([SDPBackend.MATH]):
            result = classifier._forward(ids)
        torch.cuda.synchronize()
        row = {
            "tokens": length,
            "seconds": time.perf_counter() - started,
            "peak_gb_over_loaded": (torch.cuda.max_memory_allocated() - base) / 2**30,
            "finite": bool(torch.isfinite(result).all()),
        }
        assert row["finite"] and row["peak_gb_over_loaded"] < 5
        rows.append(row)
        del ids, result
        print("math only", json.dumps(row), flush=True)
    (args.output / f"{args.model}-math-only.json").write_text(json.dumps(rows, indent=2) + "\n")


if __name__ == "__main__":
    main()
