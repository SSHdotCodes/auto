import importlib.util
import os
from pathlib import Path

import pytest


@pytest.mark.model
def test_published_weights():
    spec = importlib.util.spec_from_file_location(
        "verify_model", Path(__file__).parents[1] / "scripts/verify_model.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    report = module.verify(
        os.environ.get("AUTO_TEST_DEVICE", "cpu"), model=os.environ.get("AUTO_TEST_MODEL", "auto-0.4b-2")
    )
    assert report["passed"] == report["total"] == 24
    assert report["runtime"]["device"] == os.environ.get("AUTO_TEST_DEVICE", "cpu")
