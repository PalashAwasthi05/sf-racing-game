import * as THREE from 'three';

// Windows. Glazing in the generated world is a dark surface, so every facade read as a grid of black holes. In daylight a
// real window shows, behind a reflection that strengthens toward grazing angles, what stands just behind the glass:
// venetian blinds, roller shades or curtains in many windows, lit by the same daylight as the wall around them, and in
// the rest a room, lit through the window and dim beside the facade; now and then a room with its lights on.
//
// Each pane is found at load (its connected box of triangles) and given its own coordinates across its face, its size
// and a random identity, so every window gets its own treatment. The treatment is a diffuse layer at the glass plane:
//  blinds    slats every 2.5 cm (averaged with distance), drawn down from the head to a random depth; roller shades;
//            curtains with vertical folds, drawn in from the sides;
//  rooms     the view ray followed into a box of room size (4 m bays, 3.6 m storeys, 6 m deep) laid out along the facade
//            in world space; the wall, floor or ceiling it meets, at the brightness daylight through the glass gives it
//            (a few percent of the facade's own); a few rooms lit by their ceiling lights.
// Lit through the glass's transmission, so tinted glass shows its layer darker. The glass's own reflection stays on top.
// Nothing here refers to a place: the room grid and the treatments are generic.

const HEADER = /* glsl */`
uniform float uGlT;
varying vec3 vGlPos;varying vec3 vGlNrm;varying vec4 vGlPane;varying float vGlId;
float glHash(vec2 p){return fract(sin(dot(p,vec2(127.1,311.7)))*43758.5453);}
// the share of a pattern of frequency f (cycles per unit of x) that survives at this pixel
float glKeep(float x,float f){return 1.-smoothstep(.25,.5,fwidth(x)*f);}
vec3 glRoom(out float lit){
 lit=0.;
 vec3 N=normalize(vGlNrm);if(abs(N.y)>.5)return vec3(.05);
 vec3 V=normalize(vGlPos-cameraPosition);float facing=-dot(V,N);if(facing<=.02)return vec3(.05);
 vec3 T=normalize(vec3(-N.z,0.,N.x));
 vec3 o=vec3(dot(vGlPos.xz,T.xz),vGlPos.y,0.);vec3 d=vec3(dot(V.xz,T.xz),V.y,facing);
 vec3 cell=vec3(4.,3.6,6.);vec2 id=floor(o.xy/cell.xy);
 vec3 lo=vec3(id*cell.xy,0.),hi=vec3((id+1.)*cell.xy,cell.z);
 vec3 dd=sign(d)*max(abs(d),vec3(1e-4));vec3 tA=(lo-o)/dd,tB=(hi-o)/dd;vec3 tx=max(tA,tB);
 float t=min(min(tx.x,tx.y),tx.z);vec3 h=o+dd*t;
 float r=glHash(id+vGlId*.013),r2=fract(r*17.37);
 vec3 wallC=mix(vec3(.72,.68,.6),vec3(.55,.58,.6),r2);
 vec3 c;lit=step(.86,r);
 if(t==tx.z)c=wallC*.9;                                                          // the back wall
 else if(t==tx.y){if(dd.y>0.){c=vec3(.85,.84,.8);}else c=vec3(.32,.27,.22)*mix(.7,1.3,r2);}  // ceiling / floor
 else c=wallC*.75;                                                               // side walls
 // furniture: a band of desks and shelving at the back wall, a little darker
 if(t==tx.z&&h.y-lo.y<1.1)c*=.6;
 // daylight falls off into the room: bright by the window, dim at the back
 return c*mix(.35,1.,exp(-t*.18));
}
// a ceiling-light panel pattern for the lit rooms (emissive)
vec3 glLights(){
 vec3 N=normalize(vGlNrm);vec3 V=normalize(vGlPos-cameraPosition);float facing=-dot(V,N);if(facing<=.02||abs(N.y)>.5)return vec3(0.);
 vec3 T=normalize(vec3(-N.z,0.,N.x));vec3 o=vec3(dot(vGlPos.xz,T.xz),vGlPos.y,0.);vec3 d=vec3(dot(V.xz,T.xz),V.y,facing);
 vec3 cell=vec3(4.,3.6,6.);vec2 id=floor(o.xy/cell.xy);if(glHash(id+vGlId*.013)<.86)return vec3(0.);
 if(d.y<=0.)return vec3(.012,.011,.009);
 float tc=((id.y+1.)*cell.y-o.y)/d.y;vec3 h=o+d*tc;if(h.z>cell.z)return vec3(.012,.011,.009);
 float panel=step(.7,fract(h.x*.5))*step(.55,fract(h.z*.33));
 return vec3(1.,.95,.84)*(.03+.9*panel)*exp(-tc*.06);
}
// what stands behind this point of the glass: its colour (as a diffuse layer at the glass) and whether it is a blind
vec3 glBehind(out float lit){
 float id=vGlId;float u=vGlPane.x,v=vGlPane.y,w=max(vGlPane.z,.05),hgt=max(vGlPane.w,.05);
 float r=glHash(vec2(id,1.3)),r2=glHash(vec2(id,7.7)),r3=glHash(vec2(id,4.1));
 // a room behind the glass is lit by the daylight the window lets in: by the window, 10-20 % of the light on the facade
 vec3 room=glRoom(lit)*.12;
 if(hgt<.25)return room;                                                          // skylights and slivers
 // venetian blinds (12 %), roller shades (8 %), curtains (8 %), open (the rest); a blind hangs a little back from the
 // glass in the window's reveal, partly shaded by its head and jambs (x .8)
 float drawn=r2<.45?1.:mix(.15,.95,r2);                                           // how far down from the head
 if(r<.12){
  float covered=step(1.-drawn,v);
  float y=v*hgt;float slat=fract(y/.025);
  vec3 col=r3<.75?mix(vec3(.6,.59,.56),vec3(.5,.49,.45),r3/.75):vec3(.24,.24,.23);
  float shade=mix(.78+.22*smoothstep(.0,.6,slat)*(1.-smoothstep(.8,1.,slat)),.9,1.-glKeep(y,40.));   // slats, averaged with distance
  // a tilted blind lets a little room through between the slats
  vec3 c=mix(col*shade,room,.18*glKeep(y,40.));
  lit*=1.-covered;return mix(room,c,covered);
 }
 if(r<.2){
  float covered=step(1.-drawn,v);
  vec3 col=mix(vec3(.7,.68,.62),vec3(.36,.36,.35),step(.7,r3));
  col*=1.-.25*(1.-smoothstep(.0,.03,abs(v-(1.-drawn))))*covered;                  // the bottom bar
  lit*=1.-covered;return mix(room,col,covered);
 }
 if(r<.28){
  float a=mix(.15,.5,r2),b=mix(.15,.5,r3);float covered=max(step(u,a),step(1.-b,u));
  if(r2>.8)covered=1.;
  vec3 col=r3<.4?vec3(.7,.66,.57):r3<.6?vec3(.48,.41,.33):r3<.75?vec3(.36,.14,.12):r3<.9?vec3(.17,.2,.29):vec3(.52,.53,.52);
  float x=u*w;float fold=.82+.18*sin(x/.14*6.2832);fold=mix(fold,.9,1.-glKeep(x,7.));
  lit*=1.-covered;return mix(room,col*fold,covered);
 }
 return room;
}
`;

const GLASS = /^(glass|glass_dark|glass_light|glass_blue|glass_clear)$/;
export function isGlazing(cls: string) { return GLASS.test(cls); }
// how much light the glass lets through, by class (tinted glazing shows what is behind it darker)
const TRANSMISSION: Record<string, number> = {glass: .72, glass_dark: .42, glass_light: .8, glass_blue: .6, glass_clear: .85};

/** One window of ordinary size: its horizontal axis (x, z), its extent along it and up, and its glass plane's offset
 *  across the axis (u = dot(P.xz, axis), off = dot(P.xz, (-az, ax))). */
export type GlazedWindow = {ax: number; az: number; u0: number; u1: number; y0: number; y1: number; off: number};

/** Gives every pane of the given glazing meshes its own coordinates across its face (glPane: u, v, width, height) and an
 *  identity (glId). A pane is a connected group of triangles (the generator draws each as a thin box). Returns the
 *  windows of ordinary size found (for the walls around them: weathering.ts, facademap.ts). */
export function paneAttributes(meshes: THREE.Mesh[]) {
  const P = new THREE.Vector3(); const windows: GlazedWindow[] = [];
  for (const mesh of meshes) {
    const g = mesh.geometry; if (g.attributes.glPane) continue;
    mesh.updateWorldMatrix(true, false);
    const pos = g.attributes.position, n = pos.count, idx = g.index;
    const world = new Float32Array(n * 3);
    for (let i = 0; i < n; i++) { P.fromBufferAttribute(pos, i).applyMatrix4(mesh.matrixWorld); world[3 * i] = P.x; world[3 * i + 1] = P.y; world[3 * i + 2] = P.z; }
    // vertices at one position (1 mm) are one; triangles join their vertices
    const key = new Map<string, number>(), rep = new Int32Array(n);
    for (let i = 0; i < n; i++) { const k = Math.round(world[3 * i] * 1000) + ',' + Math.round(world[3 * i + 1] * 1000) + ',' + Math.round(world[3 * i + 2] * 1000); let r = key.get(k); if (r === undefined) { r = i; key.set(k, i); } rep[i] = r; }
    const parent = new Int32Array(n); for (let i = 0; i < n; i++) parent[i] = i;
    const find = (x: number): number => { while (parent[x] !== x) { parent[x] = parent[parent[x]]; x = parent[x]; } return x; };
    const join = (a: number, b: number) => { a = find(a); b = find(b); if (a !== b) parent[b] = a; };
    const tris = idx ? idx.count : n;
    for (let t = 0; t < tris; t += 3) { const a = rep[idx ? idx.getX(t) : t], b = rep[idx ? idx.getX(t + 1) : t + 1], c = rep[idx ? idx.getX(t + 2) : t + 2]; join(a, b); join(a, c); }
    const comps = new Map<number, number[]>();
    for (let i = 0; i < n; i++) { const r = find(rep[i]); let c = comps.get(r); if (!c) { c = []; comps.set(r, c); } c.push(i); }
    // each pane: its horizontal axis (the longest horizontal extent between its vertices), its extents along it and up,
    // and its plane's offset across it
    type Pane = {verts: number[]; ax: number; az: number; umin: number; umax: number; ymin: number; ymax: number; off: number; cx: number; cy: number; cz: number};
    const panes: Pane[] = [];
    for (const [, verts] of comps) {
      let ax = 1, az = 0, best = -1; const i0 = verts[0];
      for (const i of verts) { const dx = world[3 * i] - world[3 * i0], dz = world[3 * i + 2] - world[3 * i0 + 2], d = dx * dx + dz * dz; if (d > best) { best = d; ax = dx; az = dz; } }
      const l = Math.hypot(ax, az) || 1; ax /= l; az /= l; if (ax < 0 || (ax === 0 && az < 0)) { ax = -ax; az = -az; }
      let umin = Infinity, umax = -Infinity, ymin = Infinity, ymax = -Infinity, cx = 0, cy = 0, cz = 0;
      for (const i of verts) { const u = world[3 * i] * ax + world[3 * i + 2] * az; umin = Math.min(umin, u); umax = Math.max(umax, u); ymin = Math.min(ymin, world[3 * i + 1]); ymax = Math.max(ymax, world[3 * i + 1]); cx += world[3 * i]; cy += world[3 * i + 1]; cz += world[3 * i + 2]; }
      const m = verts.length; cx /= m; cy /= m; cz /= m;
      panes.push({verts, ax, az, umin, umax, ymin, ymax, off: -cx * az + cz * ax, cx, cy, cz});
    }
    // windows: panes in one plane that touch across a mullion (gaps up to 15 cm) are one window
    const wp = new Int32Array(panes.length); for (let i = 0; i < panes.length; i++) wp[i] = i;
    const wf = (x: number): number => { while (wp[x] !== x) { wp[x] = wp[wp[x]]; x = wp[x]; } return x; };
    const grid = new Map<string, number[]>(), cellOf = (x: number, z: number) => Math.floor(x / 3) + ',' + Math.floor(z / 3);
    panes.forEach((p, i) => { const k = cellOf(p.cx, p.cz); const c = grid.get(k); if (c) c.push(i); else grid.set(k, [i]); });
    panes.forEach((p, i) => {
      const gx = Math.floor(p.cx / 3), gz = Math.floor(p.cz / 3);
      for (let dx = -1; dx <= 1; dx++) for (let dz = -1; dz <= 1; dz++) for (const j of grid.get((gx + dx) + ',' + (gz + dz)) ?? []) {
        if (j <= i) continue; const q = panes[j];
        if (Math.abs(p.ax * q.ax + p.az * q.az) < .985 || Math.abs(p.off - (-q.cx * p.az + q.cz * p.ax)) > .25) continue;
        const qc = q.cx * p.ax + q.cz * p.az, qu0 = qc - (q.umax - q.umin) / 2, qu1 = qc + (q.umax - q.umin) / 2;   // q's extent along p's axis
        if (q.ymin > p.ymax + .15 || p.ymin > q.ymax + .15 || qu0 > p.umax + .15 || p.umin > qu1 + .15) continue;
        const a = wf(i), b = wf(j); if (a !== b) wp[b] = a;
      }
    });
    const groups = new Map<number, number[]>(); panes.forEach((_, i) => { const r = wf(i); const g0 = groups.get(r); if (g0) g0.push(i); else groups.set(r, [i]); });
    const pane = new Float32Array(n * 4), ident = new Float32Array(n);
    const hash = (a: number, b: number, c: number) => (Math.abs(Math.sin(a * 12.9898 + b * 78.233 + c * 37.719) * 43758.5453)) % 1 * 1000;
    for (const [, members] of groups) {
      const p0 = panes[members[0]]; let umin = Infinity, umax = -Infinity, ymin = Infinity, ymax = -Infinity, cx = 0, cy = 0, cz = 0;
      for (const k of members) { const p = panes[k]; for (const i of p.verts) { const u = world[3 * i] * p0.ax + world[3 * i + 2] * p0.az; umin = Math.min(umin, u); umax = Math.max(umax, u); ymin = Math.min(ymin, world[3 * i + 1]); ymax = Math.max(ymax, world[3 * i + 1]); } cx += p.cx; cy += p.cy; cz += p.cz; }
      const wid = hash(cx / members.length, cy / members.length, cz / members.length);
      // a window of ordinary size takes one treatment across all its panes; a curtain wall (one great connected glazing)
      // takes one per room-sized cell (4 m bays, 3.6 m storeys), each pane across its own face
      const ordinary = umax - umin <= 4.5 && ymax - ymin <= 4.2;
      if (ordinary && umax - umin > .2 && ymax - ymin > .3) windows.push({ax: p0.ax, az: p0.az, u0: umin, u1: umax, y0: ymin, y1: ymax, off: p0.off});
      for (const k of members) {
        const p = panes[k];
        const pu0 = ordinary ? umin : Math.min(...p.verts.map(i => world[3 * i] * p0.ax + world[3 * i + 2] * p0.az)), pu1 = ordinary ? umax : Math.max(...p.verts.map(i => world[3 * i] * p0.ax + world[3 * i + 2] * p0.az));
        const py0 = ordinary ? ymin : p.ymin, py1 = ordinary ? ymax : p.ymax, w = Math.max(pu1 - pu0, 1e-3), h = Math.max(py1 - py0, 1e-3);
        const id = ordinary ? wid : hash(wid, Math.floor(p.ymin / 3.6), Math.floor((p.cx * p0.ax + p.cz * p0.az) / 4));
        for (const i of p.verts) { const u = world[3 * i] * p0.ax + world[3 * i + 2] * p0.az; pane[4 * i] = (u - pu0) / w; pane[4 * i + 1] = (world[3 * i + 1] - py0) / h; pane[4 * i + 2] = w; pane[4 * i + 3] = h; ident[i] = id; }
      }
    }
    g.setAttribute('glPane', new THREE.BufferAttribute(pane, 4)); g.setAttribute('glId', new THREE.BufferAttribute(ident, 1));
  }
  return windows;
}

/** Draws what stands behind each given glazing material's panes (chained after any existing onBeforeCompile). `cls` names
 *  each material's class, for its transmission. The meshes must carry paneAttributes(). */
export function glazeMaterials(materials: Iterable<THREE.MeshStandardMaterial>, cls?: Map<THREE.Material, string>) {
  for (const m of materials) {
    if (m.userData.glazed) continue; m.userData.glazed = true;
    const T = TRANSMISSION[cls?.get(m) ?? 'glass'] ?? .72;
    const previous = m.onBeforeCompile.bind(m);
    m.onBeforeCompile = (shader, renderer) => {
      previous(shader, renderer);
      shader.uniforms.uGlT = {value: T};
      shader.vertexShader = shader.vertexShader.replace('#include <common>', '#include <common>\nattribute vec4 glPane;attribute float glId;varying vec3 vGlPos;varying vec3 vGlNrm;varying vec4 vGlPane;varying float vGlId;')
        .replace('#include <project_vertex>', '#include <project_vertex>\nvGlPos=(modelMatrix*vec4(transformed,1.)).xyz;vGlNrm=normalize(mat3(modelMatrix)*objectNormal);vGlPane=glPane;vGlId=glId;');
      shader.fragmentShader = shader.fragmentShader.replace('#include <common>', '#include <common>\n' + HEADER)
        // the layer behind the glass, lit as a diffuse surface at the glass plane by the light reaching the facade, through
        // the glass's transmission; a room shows a few percent of it (daylight through a window into a room)
        .replace('#include <map_fragment>', '#include <map_fragment>\nfloat glLit;vec3 glLayer=glBehind(glLit);{float f=clamp(-dot(normalize(vGlPos-cameraPosition),normalize(vGlNrm)),0.,1.);float F=.15+.85*pow(1.-f,5.);diffuseColor.rgb=glLayer*uGlT*(1.-F);}')
        .replace('#include <emissivemap_fragment>', '#include <emissivemap_fragment>\ntotalEmissiveRadiance+=glLights()*uGlT*glLit;')
        // coated double glazing reflects about 15 % of the light face-on (one plain pane 4 %)
        .replace('#include <lights_physical_fragment>', THREE.ShaderChunk.lights_physical_fragment.replace('mix( vec3( 0.04 ), diffuseColor.rgb, metalnessFactor )', 'mix( vec3( 0.15 ), diffuseColor.rgb, metalnessFactor )'));
    };
    const key = m.customProgramCacheKey.bind(m);
    m.customProgramCacheKey = () => key() + '|glazed2';
    m.needsUpdate = true;
  }
}
