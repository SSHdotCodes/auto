#!/usr/bin/env python3
"""Dependency-free, isolated installer. Never executes content submitted to the Auto model."""

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
import venv
from pathlib import Path

VERSION = "0.1.0"
BUILDS = {
    "cpu": ("2.13.0", "https://download.pytorch.org/whl/cpu"),
    "mps": ("2.13.0", None),
    "cuda": ("2.13.0", "https://download.pytorch.org/whl/cu130"),
    "cuda12": ("2.8.0", "https://download.pytorch.org/whl/cu128"),
    "cuda-legacy": ("2.7.1", "https://download.pytorch.org/whl/cu118"),
    "rocm": ("2.13.0", "https://download.pytorch.org/whl/rocm7.1"),
}


def choose_backend(system, machine, nvidia=None, rocm=False):
    if system == "Darwin":
        return "mps" if machine.lower() in {"arm64", "aarch64"} else "cpu"
    if nvidia:
        capability, driver = nvidia
        if capability >= 7.5 and driver >= 580:
            return "cuda"
        if capability >= 7.5 and driver >= 570:
            return "cuda12"
        if 5 <= capability < 10 and driver >= 520:
            return "cuda-legacy"
        return "cpu"
    if system == "Linux" and machine.lower() in {"x86_64", "amd64"} and rocm:
        return "rocm"
    return "cpu"


def detect_backend():
    nvidia = None
    if shutil.which("nvidia-smi"):
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=compute_cap,driver_version", "--format=csv,noheader"],
                text=True,
                capture_output=True,
                check=True,
                timeout=10,
            )
            cc, driver = result.stdout.splitlines()[0].split(",")
            nvidia = (float(cc), int(driver.strip().split(".")[0]))
        except (OSError, ValueError, subprocess.SubprocessError):
            pass
    rocm = Path("/opt/rocm").exists() or bool(shutil.which("rocminfo"))
    return choose_backend(platform.system(), platform.machine(), nvidia, rocm)


def install_root():
    if os.environ.get("AUTO_INSTALL_DIR"):
        return Path(os.environ["AUTO_INSTALL_DIR"]).expanduser().resolve()
    if os.name == "nt":
        return Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData/Local")) / "Auto/install"
    if platform.system() == "Darwin":
        return Path.home() / "Library/Application Support/Auto/install"
    return Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local/share")) / "auto-install"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--agent", choices=["pi", "opencode", "hermes", "all"], default="pi")
    parser.add_argument("--backend", choices=["auto", *BUILDS], default="auto")
    parser.add_argument(
        "--source", default=f"https://github.com/SSHDotCodes/auto/archive/refs/tags/v{VERSION}.zip"
    )
    parser.add_argument("--no-start", action="store_true")
    parser.add_argument("--no-path", action="store_true", help="Do not create a user CLI symlink")
    parser.add_argument(
        "--plan", action="store_true", help="Print the installation plan without changing files"
    )
    args = parser.parse_args()
    if not (3, 10) <= sys.version_info[:2] < (3, 14):
        parser.error("Use Python 3.10–3.13; the shell and PowerShell installers provide Python 3.13")
    backend = detect_backend() if args.backend == "auto" else args.backend
    root = install_root()
    if args.plan:
        print(
            json.dumps(
                {"directory": str(root), "backend": backend, "torch": BUILDS[backend], "agent": args.agent}
            )
        )
        return
    if platform.system() == "Darwin" and platform.machine().lower() not in {"arm64", "aarch64"}:
        parser.error(
            "This release requires Apple Silicon on macOS: current PyTorch wheels no longer support Intel Macs. An x86-64 Linux VM can run the CPU build."
        )
    if args.backend == "rocm" and platform.system() != "Linux":
        parser.error("The ROCm installer targets supported Linux GPUs. Choose --backend cpu on Windows.")
    if root.exists() and any(root.iterdir()) and not (root / "auto-installation.json").exists():
        parser.error(f"Refusing to reuse an unowned installation directory: {root}")
    root.mkdir(parents=True, exist_ok=True)
    (root / "auto-installation.json").write_text(
        json.dumps({"project": "SSHDotCodes/auto", "version": VERSION})
    )
    env = root / "venv"
    python = env / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    uv = shutil.which("uv")
    if not python.exists():
        if uv:
            subprocess.run([uv, "venv", "--python", sys.executable, str(env)], check=True)
        else:
            venv.EnvBuilder(with_pip=True).create(env)
    pip = [uv, "pip", "install", "--python", str(python)] if uv else [str(python), "-m", "pip", "install"]
    version, index = BUILDS[backend]
    print(f"Installing Auto with the {backend} build in {root}", flush=True)
    command = [*pip, "torch==" + version]
    if index:
        command += ["--index-url", index]
    try:
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError:
        if args.backend != "auto" or backend == "cpu":
            raise
        print("Accelerator wheel unavailable for this platform; installing CPU support.", flush=True)
        backend = "cpu"
        version, index = BUILDS[backend]
        subprocess.run([*pip, "torch==" + version, "--index-url", index], check=True)
    constraint = root / "torch-constraint.txt"
    constraint.write_text("torch==" + version + "\n")
    subprocess.run([*pip, "--constraint", str(constraint), args.source], check=True)
    command = [str(python), "-m", "auto_gate", "install", args.agent]
    if args.no_start:
        command += ["--no-start"]
    subprocess.run(command, check=True)
    executable = env / ("Scripts/auto.exe" if os.name == "nt" else "bin/auto")
    if os.name != "nt" and not args.no_path:
        bin_dir = Path.home() / ".local/bin"
        bin_dir.mkdir(parents=True, exist_ok=True)
        link = bin_dir / "auto"
        if not link.exists() or (link.is_symlink() and link.resolve() == executable.resolve()):
            if link.is_symlink():
                link.unlink()
            link.symlink_to(executable)
    print(
        f"\nInstalled. Restart {args.agent}. CLI: {executable}\nRun that executable with doctor to see your active backend."
    )


if __name__ == "__main__":
    main()
