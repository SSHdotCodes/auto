import sharp from "sharp";
import {
  copyFile,
  mkdir,
  readdir,
  readFile,
  writeFile,
} from "node:fs/promises";
import { fileURLToPath } from "node:url";
import path from "node:path";

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const site = path.join(root, "site");
await mkdir(path.join(site, "assets"), { recursive: true });
await Promise.all([
  // Social preview, rendered by scripts/social_images.py; the PNG source is not served.
  sharp(path.join(site, "assets/og.png"))
    .webp({ quality: 92 })
    .toFile(path.join(site, "assets/og.webp")),
  ...["install.sh", "install.ps1", "install.py"].map((name) =>
    copyFile(path.join(root, "scripts", name), path.join(site, name)),
  ),
]);
for (const [pkg, match, target] of [
  [
    "@fontsource-variable/geist",
    "geist-latin-wght-normal.woff2",
    "geist.woff2",
  ],
  [
    "@fontsource-variable/geist-mono",
    "geist-mono-latin-wght-normal.woff2",
    "geist-mono.woff2",
  ],
  [
    "@fontsource/instrument-serif",
    "instrument-serif-latin-400-italic.woff2",
    "instrument-serif-italic.woff2",
  ],
]) {
  const files = path.join(root, "node_modules", pkg, "files");
  if (!(await readdir(files)).includes(match))
    throw new Error(`Missing font: ${match}`);
  await copyFile(path.join(files, match), path.join(site, "assets", target));
  await copyFile(
    path.join(root, "node_modules", pkg, "LICENSE"),
    path.join(site, "assets", target + ".LICENSE"),
  );
}
const probes = JSON.parse(
  await readFile(path.join(root, "tests/fixtures/probes.json"), "utf8"),
);
await writeFile(
  path.join(site, "probes.json"),
  JSON.stringify(probes, null, 2) + "\n",
);
console.log("Built social image, fonts, installers, and probe fixtures.");
