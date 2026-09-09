import sharp from 'sharp';
import {copyFile, mkdir, readdir, readFile, writeFile} from 'node:fs/promises';
import {fileURLToPath} from 'node:url';
import path from 'node:path';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const site = path.join(root, 'site');
await mkdir(path.join(site, 'assets'), {recursive: true});
await Promise.all([
  sharp(path.join(site, 'assets/mountain.png')).webp({quality: 90}).toFile(path.join(site, 'assets/mountain.webp')),
  sharp(path.join(site, 'assets/mountain.png')).resize({width: 900}).webp({quality: 86}).toFile(path.join(site, 'assets/mountain-mobile.webp')),
  ...['install.sh', 'install.ps1', 'install.py'].map(name => copyFile(path.join(root, 'scripts', name), path.join(site, name))),
]);
for (const [pkg, match, target] of [
  ['@fontsource-variable/manrope', 'manrope-latin-wght-normal.woff2', 'manrope.woff2'],
  ['@fontsource/instrument-serif', 'instrument-serif-latin-400-italic.woff2', 'instrument-serif-italic.woff2'],
]) {
  const files = path.join(root, 'node_modules', pkg, 'files');
  if (!(await readdir(files)).includes(match)) throw new Error(`Missing font: ${match}`);
  await copyFile(path.join(files, match), path.join(site, 'assets', target));
  await copyFile(path.join(root, 'node_modules', pkg, 'LICENSE'), path.join(site, 'assets', target + '.LICENSE'));
}
const probes = JSON.parse(await readFile(path.join(root, 'tests/fixtures/probes.json'), 'utf8'));
await writeFile(path.join(site, 'probes.json'), JSON.stringify(probes, null, 2) + '\n');
console.log('Built local image, fonts, installers, and probe fixtures.');
