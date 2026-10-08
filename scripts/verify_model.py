"""Reproducible real-weight smoke checks; does not execute any proposed tool calls."""

import argparse
import json
import platform
import statistics
import time
from pathlib import Path

from auto_gate.config import DEFAULT_MODEL, MODELS, Settings
from auto_gate.model import Classifier, download
from auto_gate.schema import ScoreRequest


def verify(device, limit=24, model=DEFAULT_MODEL):
    settings = Settings(device=device, model=model)
    download(settings)
    started = time.perf_counter()
    model = Classifier(settings)
    load_seconds = time.perf_counter() - started
    fixtures = json.loads((Path(__file__).parents[1] / "tests/fixtures/probes.json").read_text())[:limit]
    results = []
    for case in fixtures:
        request = ScoreRequest(**{key: case[key] for key in ("user_request", "call", "history")})
        started = time.perf_counter()
        result = model.score(request).model_dump(exclude_none=True)
        results.append(
            {
                "id": case["id"],
                "expected": case["expected"],
                **result,
                "seconds": time.perf_counter() - started,
            }
        )
    runtime = model.info()
    model.settings.max_tokens = 128
    too_long = model.score(
        ScoreRequest(
            user_request="Read the README. " * 200, call={"tool": "read", "args": {"path": "README.md"}}
        )
    )
    incomplete = model.score(
        ScoreRequest(
            user_request="Read the README.",
            context_complete=False,
            call={"tool": "read", "args": {"path": "README.md"}},
        )
    )
    report = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "requested_device": device,
        "runtime": runtime,
        "load_seconds": load_seconds,
        "passed": sum(row["decision"] == row["expected"] for row in results),
        "total": len(results),
        "median_seconds": statistics.median(row["seconds"] for row in results),
        "over_budget": too_long.decision,
        "missing_context": incomplete.decision,
        "results": results,
    }
    assert too_long.decision == incomplete.decision == "review"
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu", choices=["cpu", "mps", "cuda"])
    parser.add_argument("--limit", type=int, default=24)
    parser.add_argument("--model", default=DEFAULT_MODEL, choices=list(MODELS))
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    report = verify(args.device, args.limit, args.model)
    Path(args.output).write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key != "results"}, indent=2))
    if report["passed"] != report["total"]:
        raise SystemExit("Real-model smoke check failed; inspect the recorded decisions")
