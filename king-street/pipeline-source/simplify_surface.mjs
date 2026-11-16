// Offline only. The embedded MIT-licensed simplifier is the upstream meshoptimizer 0.25.0 release.
// Binary protocol: one mesh at a time so packaging a city doesn't retain a second city in memory.
import {readSync, writeSync} from 'node:fs';
import {MeshoptSimplifier} from './vendor/meshoptimizer/simplifier.mjs';
await MeshoptSimplifier.ready;
function readBytes(length, allowEof = false) {
  const bytes = Buffer.allocUnsafeSlow(length);
  let n = 0;
  while (n < length) {
    const got = readSync(0, bytes, n, length - n);
    if (!got) {
      if (allowEof && !n) return null;
      throw Error('Incomplete surface');
    }
    n += got;
  }
  return bytes.buffer;
}
function send(bytes) {
  const data = Buffer.from(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  for (let offset = 0; offset < data.length;) offset += writeSync(1, data, offset);
}
for (;;) {
  const head = readBytes(12, true);
  if (!head) break;
  const [vertices, indices, channels] = new Uint32Array(head);
  const points = new Float32Array(readBytes(vertices * 12));
  const appearance = new Float32Array(readBytes(vertices * channels * 4));
  const triangles = new Uint32Array(readBytes(indices * 4));
  // Normals protect creases. Other attributes, including UV seams, also
  // participate in the metric. Boundary loops remain pinned around openings.
  const [out] = MeshoptSimplifier.simplifyWithAttributes(triangles, points, 3,
    appearance, channels, Array(channels).fill(1), null, Math.floor(indices / 12) * 3,
    .006, ['LockBorder', 'ErrorAbsolute']);
  send(new Uint32Array([out.length]));
  send(out);
}
