import asyncio
import hmac
import os
import secrets
import socket
from concurrent.futures import ThreadPoolExecutor

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from .config import Settings, home, write_json
from .schema import ScoreRequest

MAX_BODY = 2_000_000


def create_app(classifier, token: str, shutdown=None):
    app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)
    worker = ThreadPoolExecutor(max_workers=1, thread_name_prefix="auto-inference")
    pending = set()

    @app.middleware("http")
    async def authenticate(request: Request, call_next):
        if request.headers.get("origin") or request.url.hostname not in {"127.0.0.1", "localhost"}:
            return JSONResponse({"error": "Local native clients only"}, status_code=403)
        supplied = request.headers.get("authorization", "")
        if not hmac.compare_digest(supplied.encode(), ("Bearer " + token).encode()):
            return JSONResponse({"error": "Unauthorized"}, status_code=401)
        if request.method == "POST":
            try:
                size = int(request.headers.get("content-length", "0"))
            except ValueError:
                return JSONResponse({"error": "Invalid content length"}, status_code=400)
            if size <= 0 or size > MAX_BODY:
                return JSONResponse({"error": "Request too large or length missing"}, status_code=413)
            body = await request.body()
            if len(body) > MAX_BODY:
                return JSONResponse({"error": "Request too large"}, status_code=413)
        return await call_next(request)

    @app.get("/health")
    def health():
        return {"ready": True, "protocol": 1, **classifier.info()}

    @app.post("/v1/score")
    async def score(request: ScoreRequest):
        if len(pending) >= 4:
            return JSONResponse({"decision": "review", "reason": "Auto queue is full"}, status_code=503)
        future = asyncio.get_running_loop().run_in_executor(worker, classifier.score, request)
        pending.add(future)
        future.add_done_callback(pending.discard)
        try:
            return await asyncio.wait_for(asyncio.shield(future), classifier.settings.timeout)
        except Exception:
            # No raw exception or input text leaves this process.
            return JSONResponse(
                {"decision": "review", "reason": "Auto inference unavailable"}, status_code=503
            )

    @app.post("/shutdown")
    def stop():
        if shutdown:
            shutdown()
        return {"stopping": True}

    return app


def serve():
    import uvicorn

    from .model import Classifier

    classifier = Classifier(Settings.load())
    token = secrets.token_urlsafe(32)
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    app = create_app(classifier, token, lambda: setattr(server, "should_exit", True))
    server = uvicorn.Server(uvicorn.Config(app, log_level="warning", access_log=False))
    state = home() / "runtime.json"
    write_json(state, {"pid": os.getpid(), "port": port, "token": token, "protocol": 1})
    try:
        server.run(sockets=[sock])
    finally:
        if state.exists():
            import json

            if json.loads(state.read_text())["token"] == token:
                state.unlink()
        sock.close()
