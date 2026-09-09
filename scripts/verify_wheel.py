"""Ensure distribution includes the real adapters, not only Python modules."""

import zipfile
from pathlib import Path

wheel = next(Path("dist").glob("*.whl"))
with zipfile.ZipFile(wheel) as archive:
    names = set(archive.namelist())
    for name in ["common.mjs", "pi.mjs", "opencode.mjs", "hermes.py"]:
        assert f"auto_gate/adapters/{name}" in names, name
    assert "auto_gate/__main__.py" in names
    assert not any(".auto-test" in name or "runtime.json" in name for name in names)
print("Wheel includes all four adapter resources and the Python entry point.")
