import * as THREE from 'three';

// Tree canopies. The generated trees carry a few hundred small flat leaf quads each, so a street tree read as bare
// sticks with green confetti. Each leaf quad becomes a leaf-cluster card here: a larger alpha-tested card showing
// dozens of leaves (drawn once on a canvas), turned at random around the quad's own position, so a crown keeps the
// shape and size the pipeline gave it but becomes a dense mass. Cards are shaded with normals pointing out from the
// local crown centre, so a canopy is lit as a rounded volume (sunlit side, shaded underside) instead of as scattered
// flat planes. Leaf colours stay the pipeline's three greens.

function rng(seed: number) { return () => { seed |= 0; seed = seed + 0x6d2b79f5 | 0; let t = Math.imul(seed ^ seed >>> 15, 1 | seed); t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t; return ((t ^ t >>> 14) >>> 0) / 4294967296; }; }

function leafClusterTexture() {
  const size = 256, c = document.createElement('canvas'); c.width = c.height = size; const g = c.getContext('2d')!; const r = rng(11);
  g.clearRect(0, 0, size, size);
  for (let i = 0; i < 110; i++) {
    const a = r() * Math.PI * 2, d = Math.sqrt(r()) * size * .4, x = size / 2 + Math.cos(a) * d, y = size / 2 + Math.sin(a) * d;
    const len = 10 + r() * 9, wid = len * (.38 + r() * .14), rot = r() * Math.PI * 2, l = 70 + r() * 24;
    g.save(); g.translate(x, y); g.rotate(rot);
    g.fillStyle = `hsl(${88 + r() * 22},${28 + r() * 16}%,${l}%)`;
    g.beginPath(); g.moveTo(-len, 0); g.quadraticCurveTo(0, -wid, len, 0); g.quadraticCurveTo(0, wid, -len, 0); g.fill();
    g.strokeStyle = `hsla(95,30%,${l - 18}%,.6)`; g.lineWidth = 1; g.beginPath(); g.moveTo(-len * .9, 0); g.lineTo(len * .9, 0); g.stroke();
    g.restore();
  }
  const t = new THREE.CanvasTexture(c); t.colorSpace = THREE.SRGBColorSpace; t.anisotropy = 4; t.generateMipmaps = true; t.minFilter = THREE.LinearMipmapLinearFilter;
  return t;
}

/** Builds one canopy mesh from the given leaf meshes (world-space triangles of small leaf quads). */
export function buildFoliage(leafMeshes: THREE.Mesh[]) {
  const centroids: number[] = [], tints: THREE.Color[] = [];
  for (const mesh of leafMeshes) {
    mesh.updateWorldMatrix(true, false);
    const g = mesh.geometry, p = g.attributes.position, idx = g.index, v = new THREE.Vector3(), s = new THREE.Vector3();
    const colour = ((mesh.material as THREE.MeshStandardMaterial).color || new THREE.Color('#557c38')).clone();
    const count = idx ? idx.count : p.count;
    for (let t = 0; t + 2 < count; t += 6) {       // one card per leaf quad (two triangles)
      s.set(0, 0, 0);
      for (let k = 0; k < 3; k++) { v.fromBufferAttribute(p, idx ? idx.getX(t + k) : t + k).applyMatrix4(mesh.matrixWorld); s.add(v); }
      s.multiplyScalar(1 / 3); centroids.push(s.x, s.y, s.z); tints.push(colour);
    }
  }
  const n = centroids.length / 3;
  // local crown centre: the mean leaf position within 2.5 m (1 m cells)
  const cells = new Map<string, number[]>(); const key = (x: number, z: number) => Math.floor(x) + ',' + Math.floor(z);
  for (let i = 0; i < n; i++) { const k = key(centroids[3 * i], centroids[3 * i + 2]); const c = cells.get(k); if (c) { c[0] += centroids[3 * i]; c[1] += centroids[3 * i + 1]; c[2] += centroids[3 * i + 2]; c[3]++; } else cells.set(k, [centroids[3 * i], centroids[3 * i + 1], centroids[3 * i + 2], 1]); }
  const centreOf = (x: number, z: number, out: THREE.Vector3) => {
    let sx = 0, sy = 0, sz = 0, c = 0; const ix = Math.floor(x), iz = Math.floor(z);
    for (let dx = -2; dx <= 2; dx++) for (let dz = -2; dz <= 2; dz++) { const cell = cells.get((ix + dx) + ',' + (iz + dz)); if (cell) { sx += cell[0]; sy += cell[1]; sz += cell[2]; c += cell[3]; } }
    return out.set(sx / c, sy / c, sz / c);
  };
  const pos = new Float32Array(n * 12), nrm = new Float32Array(n * 12), card = new Float32Array(n * 12), uv = new Float32Array(n * 8), col = new Float32Array(n * 12), index = new Uint32Array(n * 6);
  const r = rng(5), p = new THREE.Vector3(), c = new THREE.Vector3(), radial = new THREE.Vector3(), d = new THREE.Vector3(), a = new THREE.Vector3(), b = new THREE.Vector3(), up = new THREE.Vector3(0, 1, 0), q = new THREE.Vector3(), tint = new THREE.Color();
  for (let i = 0; i < n; i++) {
    p.set(centroids[3 * i], centroids[3 * i + 1], centroids[3 * i + 2]); centreOf(p.x, p.z, c);
    radial.copy(p).sub(c); if (radial.lengthSq() < 1e-4) radial.set(r() - .5, .5, r() - .5); radial.normalize();
    d.set(r() * 2 - 1, r() * 2 - 1, r() * 2 - 1).normalize().multiplyScalar(.8).addScaledVector(radial, .6).normalize();   // card facing
    a.crossVectors(d, Math.abs(d.y) > .9 ? new THREE.Vector3(1, 0, 0) : up).normalize(); b.crossVectors(d, a).normalize();
    const size = 1.05 + r() * .6, rot = r() * Math.PI * 2, ca = Math.cos(rot), sa = Math.sin(rot);
    q.copy(a).multiplyScalar(ca).addScaledVector(b, sa); b.copy(a).multiplyScalar(-sa).addScaledVector(b, ca); a.copy(q);
    // canopy albedo: a crown in sun is a dark, slightly olive green (about 0.05-0.1), not the bright lawn green the
    // pipeline's leaf colours times this texture gave (about 0.15-0.28, a toy tree)
    tint.copy(tints[i]).multiplyScalar(.62 + r() * .38); { const l = .2126 * tint.r + .7152 * tint.g + .0722 * tint.b; tint.lerp(new THREE.Color(l * 1.08, l, l * .78), .18); }
    const corners = [[-1, -1, 0, 0], [1, -1, 1, 0], [1, 1, 1, 1], [-1, 1, 0, 1]];
    for (let k = 0; k < 4; k++) {
      const [sx, sy, u, w] = corners[k]; const o = (4 * i + k) * 3;
      pos[o] = p.x + (a.x * sx + b.x * sy) * size / 2 + radial.x * .15; pos[o + 1] = p.y + (a.y * sx + b.y * sy) * size / 2 + radial.y * .15; pos[o + 2] = p.z + (a.z * sx + b.z * sy) * size / 2 + radial.z * .15;
      // shading normal: out from the crown centre, tipped up (a canopy is sky-lit from above)
      q.set(pos[o] - c.x, pos[o + 1] - c.y, pos[o + 2] - c.z).normalize().addScaledVector(up, .35).normalize();
      nrm[o] = q.x; nrm[o + 1] = q.y; nrm[o + 2] = q.z; col[o] = tint.r; col[o + 1] = tint.g; col[o + 2] = tint.b;
      card[o] = d.x; card[o + 1] = d.y; card[o + 2] = d.z;
      uv[(4 * i + k) * 2] = u; uv[(4 * i + k) * 2 + 1] = w;
    }
    index.set([4 * i, 4 * i + 1, 4 * i + 2, 4 * i, 4 * i + 2, 4 * i + 3], 6 * i);
  }
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute('position', new THREE.BufferAttribute(pos, 3)); geometry.setAttribute('normal', new THREE.BufferAttribute(nrm, 3));
  geometry.setAttribute('uv', new THREE.BufferAttribute(uv, 2)); geometry.setAttribute('color', new THREE.BufferAttribute(col, 3)); geometry.setAttribute('cardNormal', new THREE.BufferAttribute(card, 3)); geometry.setIndex(new THREE.BufferAttribute(index, 1));
  geometry.computeBoundingSphere(); geometry.computeBoundingBox();
  const material = new THREE.MeshStandardMaterial({ map: leafClusterTexture(), vertexColors: true, alphaTest: .45, side: THREE.DoubleSide, roughness: .85, metalness: 0 });
  material.alphaToCoverage = true; material.name = 'foliage';
  const mesh = new THREE.Mesh(geometry, material); mesh.name = 'vegetation___foliage'; mesh.castShadow = true; mesh.receiveShadow = true; mesh.userData.layer = 'vegetation';
  return mesh;
}

/** Leaf cards turned edge-on to the view fade out (they smeared into streaks), and the leaf alpha is sharpened with the
 *  texture's mip level so distant crowns stay crisp leaves instead of blurred blotches. Chained after the shadow
 *  cascades' onBeforeCompile, so call it after they have set the material up. */
export function finishFoliage(material: THREE.MeshStandardMaterial) {
  const previous = material.onBeforeCompile.bind(material);
  material.onBeforeCompile = (shader, renderer) => {
    previous(shader, renderer);
    shader.vertexShader = shader.vertexShader.replace('#include <common>', '#include <common>\nattribute vec3 cardNormal;varying vec3 vCardN;varying vec3 vCardPos;')
      .replace('#include <project_vertex>', '#include <project_vertex>\nvCardN=normalize(mat3(modelMatrix)*cardNormal);vCardPos=(modelMatrix*vec4(transformed,1.)).xyz;');
    shader.fragmentShader = shader.fragmentShader.replace('#include <common>', '#include <common>\nvarying vec3 vCardN;varying vec3 vCardPos;')
      .replace('#include <map_fragment>', `#include <map_fragment>
 {vec2 g=vMapUv*256.;float lod=max(0.,log2(max(length(dFdx(g)),length(dFdy(g)))));diffuseColor.a=min(1.,diffuseColor.a*(1.+lod*.35));
  float edge=abs(dot(normalize(cameraPosition-vCardPos),normalize(vCardN)));diffuseColor.a*=smoothstep(.12,.38,edge);}`);
  };
  const key = material.customProgramCacheKey.bind(material);
  material.customProgramCacheKey = () => key() + '|foliage';
  material.needsUpdate = true;
}
