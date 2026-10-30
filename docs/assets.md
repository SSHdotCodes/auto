# Asset provenance

The page draws its starfield and horizon glow in CSS; it uses no photographs. The link preview `site/assets/og.png` is rendered from HTML by `scripts/render-social.mjs` (`node scripts/render-social.mjs og site/assets/og.png`) and encoded to `og.webp` by `scripts/build-site.mjs`; only the WebP is served. The same script renders the launch post image (`x`).

Typography: Geist and Geist Mono (variable), and Instrument Serif italic, sourced through their Fontsource npm packages. All three are distributed under the SIL Open Font License; copies are included beside the built font files in `site/assets/`. These retain their original licenses. The Auto mark is a project-created SVG. Project code is Apache-2.0.

Earlier versions used a generated mountain photograph (OpenAI ImageGen, 2026-09-09); it was removed in favor of the ssh.codes theme.
