"""Small, loopback-only static origin. Cloudflare/Pi router terminates public HTTPS."""

import argparse
import json
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlsplit

ROOT = Path(__file__).resolve().parent
PUBLIC = {
    "index.html",
    "style.css",
    "app.js",
    "probes.json",
    "install.sh",
    "install.ps1",
    "install.py",
    "robots.txt",
    "sitemap.xml",
}


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(ROOT), **kwargs)

    def send_head(self):
        route = unquote(urlsplit(self.path).path).lstrip("/") or "index.html"
        if route == "healthz":
            body = json.dumps({"status": "ok", "app": "auto", "version": "0.2.0"}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            import io

            return io.BytesIO(body)
        if route not in PUBLIC and not (
            route.startswith("assets/")
            and len(Path(route).parts) == 2
            and not Path(route).name.startswith(".")
            and Path(route).suffix in {".webp", ".svg", ".woff2", ".LICENSE"}
        ):
            self.send_error(404)
            return None
        return super().send_head()

    def end_headers(self):
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "strict-origin-when-cross-origin")
        self.send_header(
            "Content-Security-Policy",
            "default-src 'self'; img-src 'self'; font-src 'self'; style-src 'self' 'unsafe-inline'; script-src 'self'; connect-src 'self'; base-uri 'none'; frame-ancestors 'none'; form-action 'none'",
        )
        self.send_header("Permissions-Policy", "camera=(), microphone=(), geolocation=()")
        self.send_header(
            "Cache-Control", "public, max-age=3600" if self.path.startswith("/assets/") else "no-cache"
        )
        super().end_headers()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8542)
    args = parser.parse_args()
    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print(f"Auto website listening on 127.0.0.1:{args.port}", flush=True)
    server.serve_forever()
