import { exportWithDrawioDesktop } from 'file:///C:/Users/Admin/.codex/skills/drawio/scripts/runtime/desktop.js';
import { execFileSync } from 'node:child_process';

const root = 'C:/Users/Admin/Desktop/实验8去噪5.19-/gb_c1_ratio_success_20260609/results_ratio_c1_20260609';
const output = `${root}/figures/final_v20_refined_20260921`;
const inputFile = `${output}/fig1_v20_refined.drawio`;

for (const format of ['png', 'pdf']) {
  const result = await exportWithDrawioDesktop({
    inputFile,
    format,
    profile: 'final',
    scale: 3.125,
    outputFile: `${output}/fig1_v20_refined.${format}`,
    execute: (executable, args, options) => execFileSync(executable,
      format === 'pdf' ? ['--crop', ...args] : args, options),
  });
  console.log(JSON.stringify(result));
}
