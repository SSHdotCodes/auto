import importlib.util
from pathlib import Path

import pytest
import yaml

from auto_gate.install import install_agent, uninstall_agent

spec = importlib.util.spec_from_file_location("bootstrap", Path(__file__).parents[1] / "scripts/install.py")
bootstrap = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bootstrap)


@pytest.mark.parametrize(
    "system,machine,gpu,rocm,expected",
    [
        ("Darwin", "arm64", None, False, "mps"),
        ("Linux", "x86_64", (12.0, 595), False, "cuda"),
        ("Windows", "AMD64", (8.9, 575), False, "cuda12"),
        ("Linux", "x86_64", (6.1, 550), False, "cuda-legacy"),
        ("Linux", "x86_64", (3.7, 470), False, "cpu"),
        ("Linux", "x86_64", (12.0, 550), False, "cpu"),
        ("Linux", "x86_64", None, True, "rocm"),
        ("Windows", "AMD64", None, True, "cpu"),
        ("Linux", "aarch64", None, False, "cpu"),
    ],
)
def test_hardware_selection(system, machine, gpu, rocm, expected):
    assert bootstrap.choose_backend(system, machine, gpu, rocm) == expected


@pytest.mark.parametrize("agent", ["pi", "opencode", "hermes"])
def test_install_reinstall_uninstall_preserves_other_plugins(tmp_path, monkeypatch, agent):
    monkeypatch.setenv("AUTO_HOME", str(tmp_path / "auto data ü"))
    root = tmp_path / "agent space ü"
    root.mkdir()
    if agent == "hermes":
        (root / "config.yaml").write_text("model: existing\nplugins:\n  enabled: [another-plugin]\n")
    paths = install_agent(agent, root)
    assert paths == install_agent(agent, root)
    assert all(Path(p).exists() for p in paths)
    assert uninstall_agent(agent) == paths
    if agent == "hermes":
        config = yaml.safe_load((root / "config.yaml").read_text())
        assert config["model"] == "existing"
        assert config["plugins"]["enabled"] == ["another-plugin"]


def test_refuse_to_overwrite_an_unrelated_plugin(tmp_path, monkeypatch):
    monkeypatch.setenv("AUTO_HOME", str(tmp_path / "data"))
    file = tmp_path / "pi/extensions/zz-auto.ts"
    file.parent.mkdir(parents=True)
    file.write_text("// existing user code")
    with pytest.raises(ValueError, match="Refusing"):
        install_agent("pi", tmp_path / "pi")
    assert file.read_text() == "// existing user code"
