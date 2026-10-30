"""Small local client. Arguments go through stdin/loopback, never process command lines."""

import json
import os
import subprocess
import sys
import time

import httpx
from filelock import FileLock

from .config import Settings, home


def connection():
    state = json.loads((home() / "runtime.json").read_text(encoding="utf-8"))
    port = state["port"]
    if not isinstance(port, int) or not 1 <= port <= 65535 or state.get("protocol") != 1:
        raise ValueError("Invalid Auto runtime state")
    return f"http://127.0.0.1:{port}", {"Authorization": "Bearer " + state["token"]}


def health():
    try:
        url, headers = connection()
        with httpx.Client(trust_env=False, timeout=2) as client:
            response = client.get(url + "/health", headers=headers)
            response.raise_for_status()
            result = response.json()
            return result if result.get("protocol") == 1 and result.get("ready") else None
    except (OSError, ValueError, KeyError, httpx.HTTPError):
        return None


def ensure_server():
    if health():
        return
    with FileLock(str(home() / "startup.lock"), timeout=180):
        if health():
            return
        log_path = home() / "runtime.log"
        flags = (
            {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.DETACHED_PROCESS}
            if os.name == "nt"
            else {"start_new_session": True}
        )
        with open(log_path, "ab") as log:
            if os.name != "nt":
                os.chmod(log_path, 0o600)
            process = subprocess.Popen(
                [sys.executable, "-m", "auto_gate", "serve"],
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=log,
                **flags,
            )
        deadline = time.monotonic() + 150
        while time.monotonic() < deadline:
            if health():
                return
            if process.poll() is not None:
                raise RuntimeError(
                    "Auto could not start. Run `auto doctor`; details are in the local runtime log."
                )
            time.sleep(0.15)
        raise RuntimeError("Auto startup timed out; run `auto doctor` before retrying")


# Client-side fitting keeps requests under the daemon's body limit; the daemon then fits exact tokens.
FIT_BYTES = 1_500_000


def score(payload):
    try:
        from .fit import describe, merge, prefit
        from .schema import ScoreRequest

        settings = Settings.load()
        trimmed = None
        if settings.fit_context and isinstance(payload, dict):
            payload, trimmed = prefit(payload, FIT_BYTES, user_cap=480_000, max_actions=2048)
        payload = ScoreRequest.model_validate(payload).model_dump()
        ensure_server()
        url, headers = connection()
        body = json.dumps(payload, ensure_ascii=False, allow_nan=False).encode("utf-8")
        headers = {**headers, "Content-Type": "application/json"}
        with httpx.Client(trust_env=False, timeout=settings.timeout + 5) as client:
            response = client.post(url + "/v1/score", content=body, headers=headers)
            response.raise_for_status()
            result = response.json()
        if result.get("decision") not in {"approve", "deny", "review"}:
            raise ValueError("Invalid decision")
        if trimmed:
            daemon = result.get("trimmed")
            result["trimmed"] = merge(trimmed, daemon)
            if result["decision"] != "review" and not daemon:
                result["reason"] = result.get("reason", "") + describe(trimmed)
        return result
    except Exception:
        return {"decision": "review", "reason": "Auto unavailable or input invalid; host review required"}


def stop():
    if not health():
        return
    url, headers = connection()
    with httpx.Client(trust_env=False, timeout=5) as client:
        client.post(url + "/shutdown", json={}, headers=headers).raise_for_status()
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline and health():
        time.sleep(0.1)
