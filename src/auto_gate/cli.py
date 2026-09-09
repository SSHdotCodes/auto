import argparse
import json
import sys

from . import __version__
from .config import Settings, home


def main():
    # Node and Hermes use UTF-8 pipes, including on Windows with a legacy console code page.
    for stream in (sys.stdin, sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(prog="auto", description="Local tool-call decisions for coding agents")
    parser.add_argument("--version", action="version", version=__version__)
    commands = parser.add_subparsers(dest="command", required=True)
    install = commands.add_parser("install", help="Download the pinned model and install an agent adapter")
    install.add_argument("agent", choices=["pi", "opencode", "hermes", "all"])
    install.add_argument(
        "--no-start", action="store_true", help="Download and install without warming the runtime"
    )
    remove = commands.add_parser("uninstall", help="Remove only Auto's agent integration")
    remove.add_argument("agent", choices=["pi", "opencode", "hermes"])
    commands.add_parser("download", help="Download the model for offline operation")
    commands.add_parser("start", help="Start or reuse the local inference process")
    commands.add_parser("stop", help="Unload the local inference process")
    commands.add_parser("doctor", help="Show installation and active backend diagnostics")
    commands.add_parser("serve", help=argparse.SUPPRESS)
    commands.add_parser("bridge", help=argparse.SUPPRESS)
    score = commands.add_parser("score", help="Classify a JSON request from stdin or a file")
    score.add_argument("--file")
    config = commands.add_parser("configure", help="Set runtime options; restart required")
    config.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"])
    config.add_argument("--attention", choices=["chunked", "flash"])
    config.add_argument("--max-tokens", type=int)
    config.add_argument("--threshold", type=float)
    config.add_argument("--timeout", type=float)
    args = parser.parse_args()
    try:
        execute(args)
    except KeyboardInterrupt:
        raise SystemExit(130)
    except Exception as exc:
        if args.command == "bridge":
            print(json.dumps({"decision": "review", "reason": "Auto unavailable; review required"}))
        else:
            print(f"Auto: {exc}", file=sys.stderr)
            raise SystemExit(1) from None


def execute(args):
    if args.command == "serve":
        from .server import serve

        serve()
        return
    if args.command in {"bridge", "score"}:
        from .client import score
        from .server import MAX_BODY

        if getattr(args, "file", None):
            with open(args.file, encoding="utf-8") as stream:
                raw = stream.read(MAX_BODY + 1)
        else:
            raw = sys.stdin.read(MAX_BODY + 1)
        if len(raw.encode("utf-8")) > MAX_BODY:
            raise ValueError("Request exceeds the local request-size limit")
        print(json.dumps(score(json.loads(raw)), ensure_ascii=False, allow_nan=False))
        return
    from .client import ensure_server, health, stop

    settings = Settings.load()
    if args.command == "configure":
        from dataclasses import asdict

        options = asdict(settings)
        options.update({k: v for k, v in vars(args).items() if k in options and v is not None})
        Settings(**options).save()
        stop()
        print("Configuration saved. The next call will start the updated runtime.")
    elif args.command in {"download", "install"}:
        from .model import download

        settings.save()
        print("Downloading the pinned Auto model (about 0.8 GB); subsequent runs stay local.")
        download(settings)
        if args.command == "install":
            from .install import install_agent

            for agent in ["pi", "opencode", "hermes"] if args.agent == "all" else [args.agent]:
                paths = install_agent(agent)
                print(f"Installed {agent}: " + ", ".join(paths))
            if not args.no_start:
                ensure_server()
                print(json.dumps(health(), indent=2))
            print("Restart your coding agent to load Auto. Existing host permission rules remain active.")
        else:
            print("Model cached. Run auto start or auto install AGENT.")
    elif args.command == "start":
        ensure_server()
        print(json.dumps(health(), indent=2))
    elif args.command == "stop":
        stop()
        print("Auto shutdown requested.")
    elif args.command == "uninstall":
        from .install import uninstall_agent

        print(json.dumps({"removed": uninstall_agent(args.agent), "model_cache_retained": True}, indent=2))
    elif args.command == "doctor":
        import importlib.metadata

        report = {
            "version": __version__,
            "home": str(home()),
            "python": sys.version.split()[0],
            "runtime": health(),
            "model": settings.model_id,
            "revision": settings.revision,
            "config": vars(settings),
            "log": str(home() / "runtime.log"),
        }
        for package in ["torch", "transformers"]:
            try:
                report[package] = importlib.metadata.version(package)
            except importlib.metadata.PackageNotFoundError:
                report[package] = "not installed"
        print(json.dumps(report, indent=2))
