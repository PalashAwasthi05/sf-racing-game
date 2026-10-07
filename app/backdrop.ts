import * as THREE from 'three';

// A generic horizon outside the modelled area. A street looking out of the model ran into an empty white haze where the
// ground stopped: no game and no real city ends like that (from any street here one sees more city, or hills across
// water, softened by distance). Outside the model's radius only, this lays a ground plane to the horizon and blocks of
// plain buildings on a street grid, all fading into the haze. Nothing in it is observed or located: the grid takes the
// model's own street orientation, the heights and footprints are drawn from the model's own buildings, and the colours
// are a plain urban palette. It is hidden in high aerial views, where it could be mistaken for reconstruction.

function rng(seed: number) { return () => { seed |= 0; seed = seed + 0x6d2b79f5 | 0; let t = Math.imul(seed ^ seed >>> 15, 1 | seed); t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t; return ((t ^ t >>> 14) >>> 0) / 4294967296; }; }

/** The dominant direction of the given meshes' long edges, modulo 90 degrees (the street grid's orientation). */
function gridAngle(meshes: THREE.Mesh[]) {
  const bins = new Float64Array(90), a = new THREE.Vector3(), b = new THREE.Vector3();
  for (const m of meshes) {
    m.updateWorldMatrix(true, false);
    const p = m.geometry.attributes.position, idx = m.geometry.index, n = idx ? idx.count : p.count;
    for (let i = 0; i < n; i += 3) for (let k = 0; k < 3; k++) {
      a.fromBufferAttribute(p, idx ? idx.getX(i + k) : i + k).applyMatrix4(m.matrixWorld); b.fromBufferAttribute(p, idx ? idx.getX(i + (k + 1) % 3) : i + (k + 1) % 3).applyMatrix4(m.matrixWorld);
      const dx = b.x - a.x, dz = b.z - a.z, len = Math.hypot(dx, dz); if (len < 4) continue;
      const deg = ((Math.atan2(dz, dx) * 180 / Math.PI) % 90 + 90) % 90; bins[Math.floor(deg)] += len;
    }
  }
  let best = 0, bv = -1; for (let d = 0; d < 90; d++) { let s = 0; for (let k = -2; k <= 2; k++) s += bins[(d + k + 90) % 90]; if (s > bv) { bv = s; best = d; } }
  return best * Math.PI / 180;
}

const WINDOWS = /* glsl */`
varying vec3 vBdPos;varying vec3 vBdNrm;varying float vBdBase;varying float vBdId;
float bdHash(vec2 p){return fract(sin(dot(p,vec2(127.1,311.7)))*43758.5453);}
vec3 bdFacade(vec3 albedo){
 vec3 N=normalize(vBdNrm);if(abs(N.y)>.5)return albedo*mix(.55,.8,bdHash(floor(vBdPos.xz*.25)));       // roofs: darker, patchy
 vec3 T=normalize(vec3(-N.z,0.,N.x));float u=dot(vBdPos.xz,T.xz);float h=vBdPos.y-vBdBase;
 // each building its own window module: storey 3.1-4.2 m, bay 1.6-3.6 m, punched or ribbon, some blind (a warehouse)
 float r1=bdHash(vec2(vBdId,1.7)),r2=bdHash(vec2(vBdId,5.3)),r3=bdHash(vec2(vBdId,9.1));
 float sh=mix(3.1,4.2,r1),bw=mix(1.6,3.6,r2),ribbon=step(.72,r3),blind=step(r3,.12);
 float storey=floor(h/sh),fy=fract(h/sh),fx=fract(u/bw);
 float win=step(mix(.3,.22,ribbon),fy)*step(fy,mix(.8,.74,ribbon))*mix(step(mix(.22,.12,r1),fx)*step(fx,mix(.78,.88,r1)),1.,ribbon)*step(sh+.2,h)*(1.-blind);
 float shop=step(h,sh+.2)*step(.15,fy)*step(.25,fx)*(1.-blind*.7);
 float lit=step(.94,bdHash(vec2(floor(u/bw),storey+vBdId)));
 vec3 glass=mix(vec3(.03,.035,.04),vec3(.06,.065,.07),r2)+lit*vec3(.05,.045,.03);
 return mix(albedo,glass,max(win,shop*.8));
}
`;

export function buildBackdrop(opts: { roads: THREE.Mesh[]; buildings: { height: number; area?: number }[]; groundAt: (x: number, z: number) => number; radius: number; far?: number; haze: THREE.Color; setup: (m: THREE.Material) => void }) {
  const R = opts.radius, FAR = opts.far ?? 3800, r = rng(9157);
  const angle = gridAngle(opts.roads), ca = Math.cos(angle), sa = Math.sin(angle);
  // ground height outside the model: the model's edge height in each direction, easing to their mean far out
  const edge: number[] = []; for (let k = 0; k < 72; k++) { const t = k / 72 * Math.PI * 2; edge.push(opts.groundAt(Math.cos(t) * (R - 6), Math.sin(t) * (R - 6))); }
  const meanEdge = edge.reduce((s, v) => s + v, 0) / edge.length;
  const groundOut = (x: number, z: number) => { const d = Math.hypot(x, z), t = (Math.atan2(z, x) / (Math.PI * 2) + 1) % 1 * 72, i = Math.floor(t) % 72, f = t - Math.floor(t); const e = edge[i] * (1 - f) + edge[(i + 1) % 72] * f; const w = Math.exp(-Math.max(0, d - R) / 350); return e * w + meanEdge * (1 - w); };
  const group = new THREE.Group(); group.name = 'backdrop';
  // ground: a ring from just inside the model's edge to the horizon, a little under the edge so the model covers it
  const ring = new THREE.RingGeometry(R - 8, FAR, 144, 40); ring.rotateX(-Math.PI / 2);
  { const p = ring.attributes.position; for (let i = 0; i < p.count; i++) { const x = p.getX(i), z = p.getZ(i); p.setY(i, groundOut(x, z) - .3); } ring.computeVertexNormals(); }
  const groundMat = new THREE.MeshStandardMaterial({ color: new THREE.Color(.11, .11, .105), roughness: 1, metalness: 0 }); groundMat.name = 'backdrop_ground'; opts.setup(groundMat);
  const ground = new THREE.Mesh(ring, groundMat); ground.name = 'backdrop___ground'; ground.receiveShadow = true; group.add(ground);
  // blocks on the model's street grid: streets every 110-150 m, lots of 14-40 m along each block face, two rows deep;
  // further out the lots grow and thin (fewer instances where the haze takes nearly all of them)
  const heights = opts.buildings.map(b => b.height).filter(h => h > 2).sort((a, b) => a - b); const H = heights.length ? heights : [12];
  // muted: at a distance a city reads as greys and dull warm tones, never a clean saturated colour
  const palette = ['#9a9083', '#8d8c88', '#7e5d50', '#b3ada2', '#686b6e', '#93836e', '#a69d8d', '#767f83', '#8f7464'].map(c => new THREE.Color(c));
  const lots: number[][] = [];
  const blockU = 140, blockV = 100, street = 16, extent = 2400;
  for (let bu = -extent; bu < extent; bu += blockU) for (let bv = -extent; bv < extent; bv += blockV) {
    const cu = bu + blockU / 2, cv = bv + blockV / 2, cx = cu * ca - cv * sa, cz = cu * sa + cv * ca, cd = Math.hypot(cx, cz);
    if (cd < R - blockU || cd > extent) continue;
    const scale = 1 + Math.max(0, cd - R) / 900;
    for (const row of [0, 1]) {
      let u = bu + street / 2;
      while (u < bu + blockU - street / 2) {
        const w = Math.min((14 + r() * 26) * scale, bu + blockU - street / 2 - u); if (w < 8) break;
        const depth = (blockV - street) / 2 - 2 - r() * 6, vmid = bv + street / 2 + (row ? (blockV - street) / 2 + depth / 2 + 2 : depth / 2);
        const um = u + w / 2, x = um * ca - vmid * sa, z = um * sa + vmid * ca, d = Math.hypot(x, z);
        u += w + (r() < .25 ? 3 + r() * 6 : .5);
        if (d < R + 14 + Math.max(w, depth) / 2 || d > extent) continue;
        if (d > 900 && r() < Math.min(.6, (d - 900) / 2500)) continue;
        // low by the model's edge (a backdrop must not stand over the street it frames), full height 250 m out
        let h = H[Math.floor(r() * H.length)] * (.75 + r() * .5) * (.45 + .55 * Math.min(1, Math.max(0, d - R) / 250)); if (d > 700 && r() < .03) h *= 2.2;
        lots.push([x, z, w - 1, Math.max(6, depth), h, Math.floor(r() * palette.length)]);
      }
    }
  }
  const box = new THREE.BoxGeometry(1, 1, 1); box.translate(0, .5, 0);
  const mat = new THREE.MeshStandardMaterial({ color: 0xffffff, roughness: .88, metalness: 0 }); mat.name = 'backdrop_blocks';
  // the shadow cascades' setup replaces onBeforeCompile, so it runs first and this chains after it
  opts.setup(mat); const previous = mat.onBeforeCompile.bind(mat);
  const hazeU = { uBdHaze: { value: opts.haze }, uBdR: { value: R } };
  mat.onBeforeCompile = (shader, renderer) => {
    previous(shader, renderer); Object.assign(shader.uniforms, hazeU);
    shader.vertexShader = shader.vertexShader.replace('#include <common>', '#include <common>\nattribute float bdBase;varying vec3 vBdPos;varying vec3 vBdNrm;varying float vBdBase;varying float vBdId;')
      .replace('#include <project_vertex>', '#include <project_vertex>\nvBdPos=(modelMatrix*instanceMatrix*vec4(transformed,1.)).xyz;vBdNrm=normalize(mat3(modelMatrix)*mat3(instanceMatrix)*objectNormal);vBdBase=bdBase;vBdId=float(gl_InstanceID);');
    shader.fragmentShader = shader.fragmentShader.replace('#include <common>', '#include <common>\nuniform vec3 uBdHaze;uniform float uBdR;\n' + WINDOWS)
      .replace('#include <map_fragment>', '#include <map_fragment>\ndiffuseColor.rgb=bdFacade(diffuseColor.rgb);')
      // more haze than the scene's own fog: the backdrop should read as distant and vague, never as a set of boxes
      .replace('#include <fog_fragment>', '{float bd=length(vBdPos.xz-cameraPosition.xz);float hz=(1.-exp(-max(bd-60.,0.)/650.))*.82;gl_FragColor.rgb=mix(gl_FragColor.rgb,uBdHaze*(.62+.22*clamp((vBdPos.y-vBdBase)/40.,0.,1.)),hz);}\n#include <fog_fragment>');
  };
  const key = mat.customProgramCacheKey.bind(mat); mat.customProgramCacheKey = () => key() + '|backdrop-blocks';
  const blocks = new THREE.InstancedMesh(box, mat, lots.length); blocks.name = 'backdrop___blocks';
  const m = new THREE.Matrix4(), q = new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(0, 1, 0), -angle), s = new THREE.Vector3(), t = new THREE.Vector3(), base = new Float32Array(lots.length);
  lots.forEach(([x, z, w, dp, h, c], i) => { const g = groundOut(x, z) - .5; s.set(w, h + .5, dp); t.set(x, g, z); m.compose(t, q, s); blocks.setMatrixAt(i, m); blocks.setColorAt(i, palette[c]); base[i] = g + .5; });
  box.setAttribute('bdBase', new THREE.InstancedBufferAttribute(base, 1));
  blocks.instanceMatrix.needsUpdate = true; if (blocks.instanceColor) blocks.instanceColor.needsUpdate = true;
  blocks.computeBoundingSphere(); blocks.castShadow = false; blocks.receiveShadow = true; group.add(blocks);
  group.traverse(o => { o.userData.layer = 'backdrop'; });
  return {
    group, materials: [groundMat, mat], count: lots.length, angle,
    /** Hidden in high aerial views (camera more than 150 m over the ground), where it could be taken for reconstruction. */
    update(camera: THREE.Camera, groundY: number) { group.visible = camera.position.y - groundY < 150; },
    dispose() { ring.dispose(); box.dispose(); groundMat.dispose(); mat.dispose(); blocks.dispose(); },
  };
}
