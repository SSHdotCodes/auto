import argparse
import json
import sys
from dataclasses import asdict

from . import __version__
from .config import MODELS, Settings, home
from .install import AGENTS

MAX_INPUT = 64 * 1024 * 1024


def main():
    # Node and Hermes use UTF-8 pipes, including on Windows with a legacy console code page.
    for stream in (sys.stdin, sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(prog="auto", description="Local tool-call decisions for coding agents")
    parser.add_argument("--version", action="version", version=__version__)
    commands = parser.add_subparsers(dest="command", required=True)
    install = commands.add_parser(
        "install",
        help="Download the pinned model and connect Auto to your agents (default: every agent found)",
    )
    install.add_argument(
        "agent", nargs="*", metavar="AGENT", help="one or more of: " + ", ".join(AGENTS) + ", all"
    )
    install.add_argument(
        "--no-start", action="store_true", help="Download and install without warming the runtime"
    )
    install.add_argument("--model", choices=list(MODELS), help="Auto model to use (default: keep current)")
    remove = commands.add_parser("uninstall", help="Remove only Auto's agent integration")
    remove.add_argument("agent", choices=AGENTS)
    download = commands.add_parser("download", help="Download the model for offline operation")
    download.add_argument("--model", choices=list(MODELS), help="Auto model to use (default: keep current)")
    commands.add_parser("start", help="Start or reuse the local inference process")
    commands.add_parser("stop", help="Unload the local inference process")
    commands.add_parser("doctor", help="Show installation and active backend diagnostics")
    commands.add_parser("serve", help=argparse.SUPPRESS)
    commands.add_parser("bridge", help=argparse.SUPPRESS)
    hook = commands.add_parser("hook", help=argparse.SUPPRESS)
    hook.add_argument("agent")
    hook.add_argument("--event")
    hook.add_argument("--home")
    score = commands.add_parser("score", help="Classify a JSON request from stdin or a file")
    score.add_argument("--file")
    config = commands.add_parser("configure", help="Set runtime options; restart required")
    config.add_argument("--model", choices=list(MODELS))
    config.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"])
    config.add_argument("--attention", choices=["chunked", "flash"])
    config.add_argument("--max-tokens", type=int)
    config.add_argument("--threshold", type=float)
    config.add_argument("--timeout", type=float)
    config.add_argument(
        "--fit-context",
        choices=["on", "off"],
        help="on (default): trim the oldest context of over-budget sessions; off: return review instead",
    )
    config.add_argument(
        "--fit-tokens",
        type=int,
        help="context size long sessions are fitted to (0: automatic, 32768 on CUDA/ROCm, 8192 otherwise)",
    )
    args, extra = parser.parse_known_args()
    # A host that runs hook commands without a shell passes the `|| exit 2` suffix as arguments; ignore it there.
    if extra and args.command != "hook":
        parser.error("unrecognized arguments: " + " ".join(extra))
    unknown = (
        sorted(set(getattr(args, "agent", None) or []) - {*AGENTS, "all"})
        if args.command == "install"
        else []
    )
    if unknown:
        parser.error(f"unknown agent {', '.join(unknown)}; choose from {', '.join(AGENTS)}, all")
    try:
        execute(args)
    except KeyboardInterrupt:
        raise SystemExit(130)
    except Exception as exc:
        if args.command == "hook":
            # Exit 2 blocks the call in hook hosts; never let a crash read as approval.
            print("Auto hook failed; call blocked. Run `auto doctor`.", file=sys.stderr)
            raise SystemExit(2) from None
        if args.command == "bridge":
            print(json.dumps({"decision": "review", "reason": "Auto unavailable; review required"}))
        else:
            print(f"Auto: {exc}", file=sys.stderr)
            raise SystemExit(1) from None


def execute(args):
    if args.command == "hook":
        if args.home:
            import os

            os.environ["AUTO_HOME"] = args.home
        from .hooks import main as hook

        raise SystemExit(hook(args.agent, args.event))
    if args.command == "serve":
        from .server import serve

        serve()
        return
    if args.command in {"bridge", "score"}:
        from .client import score
        from .server import MAX_BODY

        # Long sessions are fitted before they reach the daemon's 2 MB body limit; this caps raw input.
        limit = MAX_INPUT if Settings.load().fit_context else MAX_BODY
        if getattr(args, "file", None):
            with open(args.file, encoding="utf-8") as stream:
                raw = stream.read(limit + 1)
        else:
            raw = sys.stdin.read(limit + 1)
        if len(raw.encode("utf-8")) > limit:
            raise ValueError("Request exceeds the local request-size limit")
        print(json.dumps(score(json.loads(raw)), ensure_ascii=False, allow_nan=False))
        return
    from .client import ensure_server, health, stop

    settings = Settings.load()
    if getattr(args, "model", None) and args.command in {"download", "install"}:
        settings = Settings(**{**asdict(settings), "model": args.model})
    if args.command == "configure":
        options = asdict(settings)
        options.update({k: v for k, v in vars(args).items() if k in options and v is not None})
        if args.fit_context:
            options["fit_context"] = args.fit_context == "on"
        Settings(**options).save()
        stop()
        print("Configuration saved. The next call will start the updated runtime.")
        if args.model or args.attention == "flash":
            print("Run `auto download` while online to fetch the selected model or kernel.")
    elif args.command in {"download", "install"}:
        from .model import download

        spec = MODELS[settings.model]
        print(
            f"Downloading the pinned {settings.model} model ({spec['download']}); subsequent runs stay local."
        )
        download(settings)
        # A running daemon has its own model/settings. Reinstallation or explicit selection
        # must unload it before the next call, including when --no-start is used.
        if args.command == "install" or getattr(args, "model", None):
            stop()
        settings.save()
        if args.command == "install":
            from .install import detect_agents, install_agent

            agents = AGENTS if "all" in args.agent else list(dict.fromkeys(args.agent)) or detect_agents()
            if not agents:
                print("No supported agent found. Name one: auto install " + " | ".join(AGENTS))
            for agent in agents:
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
