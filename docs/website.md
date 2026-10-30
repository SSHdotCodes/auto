# Website

`site/` is the complete static website. Run `npm ci && npm run build:site` to encode the social preview image, copy local fonts and their licenses, and copy the canonical installer scripts. `python site/serve.py --port 8542` serves only the intended public files on loopback. A reverse proxy provides HTTPS. No model inference or management interface is exposed through the website.

The production origin is a Raspberry Pi, managed as the `auto` app. Deploy the whole built site tree, set the command to `python3 serve.py --port 8542`, enable autostart and restart-always, and route `auto.ssh.codes` to the loopback origin. `/healthz` returns a minimal JSON readiness response. Before modifying an existing deployment, pull its complete current source and reapply intended changes to a fresh pull immediately before upload.

Charts report the v3 benchmark (2,948 held-out tool calls in near-twin pairs) for the v3 models, three hosted guards and auto-0.4b-2. Metric selectors change both bars and accessible labels; all axes start at zero. Probe examples are static and clearly identified. Install controls generate agent/OS-specific commands and support copying or keyboard fallback. The site has no analytics or third-party font requests.

The site follows the ssh.codes theme: a black, CSS-only starfield and horizon glow, Geist and Geist Mono with Instrument Serif italic accents. It loads no images in the page itself. `assets/og.png`, the link preview, is rendered by `scripts/render-social.mjs` and served as WebP. See [asset provenance](assets.md) for fonts and licenses.
