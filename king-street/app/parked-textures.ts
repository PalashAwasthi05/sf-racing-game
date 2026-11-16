import * as THREE from 'three';

type Capabilities = Pick<THREE.WebGLRenderer, 'extensions'>;
/** The fixed header describes a complete BC7 chain, including 2x2/1x1 blocks. */
export function parseParkedBc7(bytes: ArrayBuffer, srgb: boolean) {
  if (bytes.byteLength < 24) throw new Error('Incomplete parked atlas');
  const header = new DataView(bytes);
  const [magic, version, width, height, levels, color] = Array.from({length: 6}, (_, i) => header.getUint32(i * 4, true));
  if (magic !== 0x37434250 || version !== 1 || color !== Number(srgb) ||
      width < 4 || height < 4 || width > 8192 || height > 8192 ||
      (width & (width - 1)) || (height & (height - 1)) || levels !== 1 + Math.log2(Math.max(width, height))) throw new Error('Invalid parked atlas');
  const mipmaps: THREE.CompressedTexture['mipmaps'] = [];
  let offset = 24;
  for (let level = 0; level < levels; level++) {
    const w = Math.max(1, width >> level), h = Math.max(1, height >> level);
    const length = Math.ceil(w / 4) * Math.ceil(h / 4) * 16;
    if (offset + length > bytes.byteLength) throw new Error('Incomplete parked atlas mip');
    mipmaps.push({data: new Uint8Array(bytes, offset, length), width: w, height: h}); offset += length;
  }
  if (offset !== bytes.byteLength) throw new Error('Unexpected parked atlas payload');
  const texture = new THREE.CompressedTexture(mipmaps, width, height, THREE.RGBA_BPTC_Format);
  texture.colorSpace = srgb ? THREE.SRGBColorSpace : THREE.NoColorSpace;
  texture.minFilter = THREE.LinearMipmapLinearFilter;
  texture.magFilter = THREE.LinearFilter;
  texture.generateMipmaps = false;
  texture.flipY = false;
  return texture;
}

export async function loadParkedAtlasPair(base: string, renderer?: Capabilities) {
  let manifest: Record<string, {file?: string}> = {};
  // Only the bundled kit ships this optional manifest; custom kits stay portable.
  if (base === '/vehicle/parked/' && renderer?.extensions.has('EXT_texture_compression_bptc')) {
    try {
      const response = await fetch(base + 'atlas-bc7.json');
      if (response.ok) manifest = await response.json();
    } catch { /* Portable assets remain available. */ }
  }
  const load = async (name: string, srgb: boolean) => {
    const file = manifest?.[name]?.file;
    if (file && /^atlas-(?:colour|surface)-[a-f0-9]{16}\.bc7$/.test(file)) {
      try {
        const response = await fetch(base + file);
        if (response.ok) return parseParkedBc7(await response.arrayBuffer(), srgb);
      } catch { /* Corrupt or unavailable companions fall back per atlas. */ }
    }
    return new THREE.TextureLoader().loadAsync(base + name + '.webp');
  };
  return Promise.all([load('atlas-colour', true), load('atlas-surface', false)]);
}
