# Contributing

Use Python 3.10–3.13 and Node 22 or newer. Install the development extra and the appropriate PyTorch wheel for your hardware. The bootstrap source can be overridden to a local checkout, but use temporary agent profiles while developing:

```sh
uv venv --python 3.13
uv pip install '.[dev]'
npm ci
uv run --no-project python -m pytest -m 'not model'
npm test
npm run build:site
```

Run `ruff check .`, `ruff format --check .`, and `python -m build` before submitting Python changes. Tests verify authorization context, attention window equivalence, plugin lifecycle, isolated install paths, local API boundaries, and installer hardware selection. `pytest -m model` downloads ~0.8 GB and tests actual CPU inference. Set `AUTO_TEST_MODEL=auto-200m-2-int4` or `auto-200m-2-int8` to test the packed checkpoints; CI checks all four variants. Use `AUTO_HOME` for an isolated cache. The 24 fixtures are illustrative checks, not a replacement for a held-out benchmark.

Browser tests run in CI through Playwright on desktop and mobile viewports. The website is a static app with local assets: `npm run build:site`, then `python site/serve.py`. No public inference endpoint is involved. Build copies the exact source installer scripts into the website.

Agent APIs move quickly. When changing an adapter, include the upstream version/commit, a contract test using the real host loader or SDK, and failure-path tests. Never make host errors silently allow tool execution. Verify that user-role text and untrusted tool outputs remain distinct. Do not discard context to fit a token budget.

Hardware contributions should include OS, architecture, driver, PyTorch version, the actual backend from `auto doctor`, and raw synthetic probe outputs. Clearly distinguish a device you ran from a compatibility inference. Do not include local bearer tokens or personal prompts.

Releases require green CI, a version bump, an annotated `vX.Y.Z` tag, built wheel/sdist assets, and matching installer version. Publish the complete built `site/` tree to your static origin. Deployment instructions for this installation are in [docs/website.md](docs/website.md).
