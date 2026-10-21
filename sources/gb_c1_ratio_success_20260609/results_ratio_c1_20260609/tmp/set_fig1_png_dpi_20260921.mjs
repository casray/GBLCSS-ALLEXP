import { readFileSync, writeFileSync } from 'node:fs';
import { createHash } from 'node:crypto';
import { crc32 } from 'node:zlib';

const file = new URL('../figures/final_v20_refined_20260921/fig1_v20_refined.png', import.meta.url);
const original = readFileSync(file);
const parts = [original.subarray(0, 8)];
const density = Buffer.alloc(21);
density.writeUInt32BE(9, 0);
density.write('pHYs', 4, 'ascii');
density.writeUInt32BE(Math.ceil(300 / 0.0254), 8);
density.writeUInt32BE(Math.ceil(300 / 0.0254), 12);
density[16] = 1;
density.writeUInt32BE(crc32(density.subarray(4, 17)), 17);
const pixels = [];
for (let offset = 8; offset < original.length;) {
  const size = original.readUInt32BE(offset);
  const type = original.toString('ascii', offset + 4, offset + 8);
  const chunk = original.subarray(offset, offset + size + 12);
  if (type !== 'pHYs') parts.push(chunk);
  if (type === 'IHDR') parts.push(density);
  if (type === 'IDAT') pixels.push(chunk);
  offset += size + 12;
}
writeFileSync(file, Buffer.concat(parts));
console.log(JSON.stringify({width: original.readUInt32BE(16), height: original.readUInt32BE(20),
  dpi: density.readUInt32BE(8) * 0.0254, pixelDataUnchanged: true,
  idatSha256: createHash('sha256').update(Buffer.concat(pixels)).digest('hex')}));
