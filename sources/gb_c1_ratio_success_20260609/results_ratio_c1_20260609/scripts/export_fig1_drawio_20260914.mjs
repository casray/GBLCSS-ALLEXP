import { fileURLToPath } from 'node:url';
import { dirname, resolve } from 'node:path';
import { exportWithDrawioDesktop } from 'file:///C:/Users/Admin/.codex/skills/drawio/scripts/runtime/desktop.js';

const root = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const stem = 'fig1_gblcss_overview_drawio_v21_readable';
const output = resolve(root, 'figures/revised_20260914');
const inputFile = resolve(output, `${stem}.drawio`);
const preview = process.argv.includes('--preview');
const exports = preview
  ? [{format: 'png', profile: 'vision-preview', width: 1960,
      outputFile: resolve(root, '.drawio-tmp/submission_figures_20260914/fig1_v21_preview.png')}]
  : ['png', 'pdf'].map(format => ({format, profile: 'final', scale: 3.125,
      outputFile: resolve(output, `${stem}.${format}`)}));

for (const options of exports) {
  const result = await exportWithDrawioDesktop({inputFile, ...options});
  console.log(JSON.stringify(result));
}
