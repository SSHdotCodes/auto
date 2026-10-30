# Website

`site/` is the complete static website. Run `npm ci && npm run build:site` to generate optimized WebP backgrounds, copy local fonts and their licenses, and copy the canonical installer scripts. `python site/serve.py --port 8542` serves only the intended public files on loopback. A reverse proxy provides HTTPS. No model inference or management interface is exposed through the website.

The production origin is a Raspberry Pi, managed as the `auto` app. Deploy the whole built site tree, set the command to `python3 serve.py --port 8542`, enable autostart and restart-always, and route `auto.ssh.codes` to the loopback origin. `/healthz` returns a minimal JSON readiness response. Before modifying an existing deployment, pull its complete current source and reapply intended changes to a fresh pull immediately before upload.

Charts report the v3 benchmark (2,948 held-out tool calls in near-twin pairs) for the v3 models, three hosted guards and auto-0.4b-2. Metric selectors change both bars and accessible labels; all axes start at zero. Probe examples are static and clearly identified. Install controls generate agent/OS-specific commands and support copying or keyboard fallback. The site has no analytics or third-party font requests.

The mountain background was generated for this project using OpenAI ImageGen on September 9, 2026. The original asset is 1672×941; WebP derivatives preserve its composition. It depicts an imagined alpine scene, not an identified real location. See [asset provenance](assets.md) for the prompt and font licenses.
