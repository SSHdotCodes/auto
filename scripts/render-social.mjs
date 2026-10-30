// Social images in the site's theme, rendered with Playwright so they use the real fonts.
//   node scripts/render-social.mjs og site/assets/og.png    (link preview, 1200×630 at 2×)
//   node scripts/render-social.mjs x  OUT.png              (launch post image, 1600×900 at 2×)
// Numbers: v3 benchmark (2,948 held-out tool calls) and median time per decision from the v0.4.0 release.
import { chromium } from "playwright";
import { fileURLToPath, pathToFileURL } from "node:url";
import path from "node:path";
import { mkdtemp, writeFile } from "node:fs/promises";
import os from "node:os";

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const font = (pkg, file) =>
  pathToFileURL(path.join(root, "node_modules", pkg, "files", file)).href;
const [kind, out] = process.argv.slice(2);
if (!["og", "x"].includes(kind) || !out)
  throw new Error("usage: render-social.mjs og|x OUT.png");

const css = `
@font-face { font-family: Geist; src: url(${font("@fontsource-variable/geist", "geist-latin-wght-normal.woff2")}); font-weight: 100 900; }
@font-face { font-family: "Geist Mono"; src: url(${font("@fontsource-variable/geist-mono", "geist-mono-latin-wght-normal.woff2")}); font-weight: 100 900; }
@font-face { font-family: Instrument; src: url(${font("@fontsource/instrument-serif", "instrument-serif-latin-400-italic.woff2")}); font-style: italic; }
* { box-sizing: border-box; margin: 0; }
body { width: var(--w); height: var(--h); background: #000; color: #f4f6fb; font-family: Geist; overflow: hidden; position: relative; }
body::before { content: ""; position: absolute; inset: 0; opacity: .6; background-image:
  radial-gradient(1px 1px at 23px 41px, #ffffffb0, transparent), radial-gradient(1px 1px at 151px 97px, #ffffff70, transparent),
  radial-gradient(1px 1px at 87px 171px, #dfeaff80, transparent), radial-gradient(1.5px 1.5px at 211px 23px, #ffffff90, transparent),
  radial-gradient(1px 1px at 301px 233px, #ffffff60, transparent), radial-gradient(1.5px 1.5px at 397px 151px, #ffffff80, transparent);
  background-size: 233px 233px, 277px 277px, 313px 313px, 389px 389px, 431px 431px, 523px 523px; }
.horizon { position: absolute; left: 50%; width: 3200px; aspect-ratio: 1; border-radius: 50%; transform: translateX(-30%);
  background: radial-gradient(circle at 50% 0%, #16335f 0%, #0a1a33 9%, #03070f 22%, #000 34%);
  box-shadow: 0 -1px 0 rgba(159,208,255,.55), 0 -18px 60px rgba(110,170,255,.28), 0 -80px 160px rgba(80,130,255,.12); }
.brand { display: flex; align-items: center; gap: 12px; font-weight: 600; letter-spacing: -.02em; }
.orb { width: 24px; height: 24px; border-radius: 50%; background: radial-gradient(circle at 34% 30%, #e2f1ff 0%, #6ea6f4 30%, #1d4c9c 60%, #04101f 85%);
  box-shadow: 0 0 0 1px rgba(159,208,255,.4), 0 0 16px rgba(110,170,255,.55); }
.mono { font-family: "Geist Mono"; text-transform: uppercase; letter-spacing: .12em; font-weight: 500; }
h1 { font-weight: 600; letter-spacing: -.05em; line-height: 1; }
em { font-family: Instrument; font-style: italic; font-weight: 400; letter-spacing: -.01em; padding-right: .06em;
  background: linear-gradient(100deg, #7597ff 0%, #95b2ff 50%, #b4ceff 100%); -webkit-background-clip: text; background-clip: text; color: transparent; }
.sky { color: #9fd0ff; }
.muted { color: rgba(226,233,247,.68); }
`;

const rows = [
  ["auto-3b-3", "open weights · 3.08B", 96.0, "21 ms", true],
  ["DeepSeek V4.1 Flash", "hosted · max reasoning", 95.6, "1.3 s", false],
  ["Codex auto-review", "hosted · medium", 95.5, "–", false],
  ["GLM-5.3 Flash", "hosted · high reasoning", 95.4, "1.2 s", false],
  ["auto-0.4b-3", "open weights · 396M", 93.1, "7 ms", true],
  ["auto-150m-3", "open weights · 150M", 88.8, "5 ms", true],
];
const lo = 85,
  hi = 100;
const pos = (v) => `${((v - lo) / (hi - lo)) * 100}%`;

const x = `
<style>${css}
:root { --w: 1600px; --h: 900px; }
.horizon { top: 868px; }
.top { position: absolute; left: 88px; right: 88px; top: 64px; display: flex; justify-content: space-between; align-items: center; }
.top .brand { font-size: 28px; }
.top .mono { font-size: 15px; }
h1 { position: absolute; left: 88px; top: 142px; font-size: 76px; }
.card { position: absolute; left: 88px; right: 88px; top: 352px; padding: 26px 40px 20px; border-radius: 28px;
  border: 1px solid rgba(255,255,255,.1); background: rgba(8,11,18,.78); backdrop-filter: blur(8px); }
.head, .row { display: grid; grid-template-columns: 340px 1fr 170px; align-items: center; column-gap: 40px; }
.head { font-size: 13px; padding-bottom: 12px; border-bottom: 1px solid rgba(255,255,255,.085); }
.head span:last-child { text-align: right; }
.row { height: 64px; border-bottom: 1px solid rgba(255,255,255,.05); }
.row:last-child { border-bottom: 0; }
.name { font-size: 25px; font-weight: 500; letter-spacing: -.02em; line-height: 1.1; }
.name small { display: block; font: 500 12.5px/1.5 "Geist Mono"; letter-spacing: .06em; text-transform: uppercase; color: rgba(226,233,247,.6); }
.track { position: relative; height: 2px; background: rgba(255,255,255,.1); border-radius: 2px; }
.fill { position: absolute; left: 0; top: 0; height: 2px; border-radius: 2px; background: rgba(255,255,255,.22); }
.auto .fill { background: linear-gradient(90deg, rgba(121,174,252,.15), #79aefc); }
.dot { position: absolute; top: 50%; width: 16px; height: 16px; margin: -8px 0 0 -8px; border-radius: 50%; background: #6b6f7a; box-shadow: 0 0 0 4px #080b12; }
.auto .dot { background: #b6dcff; box-shadow: 0 0 0 4px #080b12, 0 0 18px rgba(121,174,252,.9); }
.val { position: absolute; top: -34px; transform: translateX(-50%); font-size: 21px; font-weight: 500; letter-spacing: -.02em; color: rgba(226,233,247,.7); white-space: nowrap; }
.auto .val { color: #fff; font-weight: 600; }
.speed { text-align: right; font: 500 27px/1 "Geist Mono"; letter-spacing: -.02em; color: rgba(226,233,247,.62); }
.auto .speed { color: #9fd0ff; }
.axis { display: grid; grid-template-columns: 340px 1fr 170px; column-gap: 40px; margin-top: 6px; }
.ticks { position: relative; height: 18px; font: 500 12px/1 "Geist Mono"; color: rgba(226,233,247,.45); }
.ticks span { position: absolute; transform: translateX(-50%); }
.foot { position: absolute; left: 88px; right: 88px; bottom: 34px; display: flex; justify-content: space-between; gap: 48px; font-size: 12.5px; }
</style>
<div class="horizon"></div>
<div class="top"><div class="brand"><span class="orb"></span>auto</div><div class="mono sky">v3 · open-weight tool-call guards</div></div>
<h1>As accurate as hosted guards.<br>At <em>60×</em> the speed.</h1>
<div class="card">
  <div class="head mono muted"><span>Model</span><span>Accuracy · v3 benchmark</span><span>Per decision</span></div>
  ${rows
    .map(
      ([
        n,
        s,
        a,
        t,
        auto,
      ]) => `<div class="row ${auto ? "auto" : ""}"><div class="name">${n}<small>${s}</small></div>
    <div class="track"><div class="fill" style="width:${pos(a)}"></div><div class="dot" style="left:${pos(a)}"></div><div class="val" style="left:${pos(a)}">${a.toFixed(1)}%</div></div>
    <div class="speed">${t}</div></div>`,
    )
    .join("")}
  <div class="axis"><span></span><div class="ticks">${[85, 90, 95, 100].map((v) => `<span style="left:${pos(v)}">${v}%</span>`).join("")}</div><span></span></div>
</div>
<div class="foot mono muted"><span>2,948 held-out tool calls · Auto on one RTX PRO 6000 · hosted via API · Codex via CLI, not timed</span><span class="sky">huggingface.co/ProCreations</span></div>
`;

const og = `
<style>${css}
:root { --w: 1200px; --h: 630px; }
.horizon { top: 520px; transform: translateX(-36%); }
.brand { position: absolute; left: 72px; top: 58px; font-size: 26px; }
.eyebrow { position: absolute; left: 72px; top: 150px; font-size: 14px; }
h1 { position: absolute; left: 72px; top: 190px; font-size: 92px; }
.stats { position: absolute; left: 72px; right: 72px; top: 438px; display: flex; gap: 64px; }
.stats b { display: block; font-size: 44px; font-weight: 500; letter-spacing: -.04em; line-height: 1; }
.stats span { display: block; margin-top: 10px; font-size: 12.5px; }
.url { position: absolute; right: 72px; top: 64px; font-size: 15px; }
</style>
<div class="horizon"></div>
<div class="brand"><span class="orb"></span>auto</div>
<div class="url mono sky">auto.ssh.codes</div>
<div class="eyebrow mono sky">Local permission intelligence for coding agents</div>
<h1>A little intelligence.<br>A <em>clearer</em> path.</h1>
<div class="stats">
  <div><b>96.0%</b><span class="mono muted">v3 accuracy · auto-3b-3</span></div>
  <div><b>21 ms</b><span class="mono muted">per decision</span></div>
  <div><b>3</b><span class="mono muted">open-weight models</span></div>
</div>
`;

const [w, h] = kind === "x" ? [1600, 900] : [1200, 630];
const browser = await chromium.launch();
const page = await browser.newPage({
  viewport: { width: w, height: h },
  deviceScaleFactor: 2,
});
// Loaded from a file so the file:// fonts are same-origin.
const html = path.join(
  await mkdtemp(path.join(os.tmpdir(), "auto-social-")),
  "card.html",
);
await writeFile(
  html,
  `<!doctype html><html><head><meta charset="utf-8"></head><body>${kind === "x" ? x : og}</body></html>`,
);
await page.goto(pathToFileURL(html).href);
await page.evaluate(() => document.fonts.ready);
const families = await page.evaluate(() =>
  [...document.fonts].filter((f) => f.status === "loaded").map((f) => f.family),
);
if (families.length < 3) throw new Error(`fonts did not load: ${families}`);
await page.screenshot({ path: out });
await browser.close();
console.log(`wrote ${out}`);
