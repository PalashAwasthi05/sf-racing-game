import * as THREE from 'three';
import {finishStaticSteps,finishStaticStepsAsync} from './static-detail-geometry.ts';

// Pane-local room depth is an approximation. Explicitly modeled glazing and
// source-observed partitions remain in the world geometry.
const HEADER = /* glsl */`
uniform float uGlT, uGlNight;
varying vec3 vGlPos;
varying vec3 vGlNrm;
varying vec4 vGlPane;
flat varying float vGlId;
varying vec2 vGlAxis;
float glHash(float p){return fract(sin(p * 12.9898) * 43758.5453);}
// Slab intersection against a scene element inside the room. Keep the nearest
// real surface, so a console edge occludes the far wall at the correct depth.
void glSolid(vec3 origin, vec3 inv, vec3 lo, vec3 hi, vec3 finish,
             inout float nearest, inout vec3 normalHit, inout vec3 tone){
 vec3 a=(lo-origin)*inv, b=(hi-origin)*inv;
 vec3 enter=min(a,b), leave=max(a,b);
 float t=max(enter.x,max(enter.y,enter.z));
 if(t<=.001 || t>=nearest || t>min(leave.x,min(leave.y,leave.z)))return;
 vec3 face=enter.x==t?vec3(1.,0.,0.):(enter.y==t?vec3(0.,1.,0.):vec3(0.,0.,1.));
 normalHit=-sign(inv)*face;
 nearest=t;
 tone=finish*(.81+.19*max(0.,normalHit.y));
}
vec3 glRoom(out vec3 lamp, out float transmission){
 lamp=vec3(0.);
 transmission=0.;
 vec3 N=normalize(vGlNrm)*(gl_FrontFacing?1.:-1.), ray=normalize(vGlPos-cameraPosition);
 float facing=-dot(ray,N);
 if(abs(N.y)>.5 || facing<=.01) return vec3(.025);
 // Two dielectric interfaces reflect light. The room disappears at grazing angles.
 transmission=uGlT*.92*(1.-pow(1.-clamp(facing,0.,1.),5.));
 float w=max(vGlPane.z,.05), h=max(vGlPane.w,.05);
 float seed=mod(vGlId,2048.);
 float finish=glHash(seed+.71);
 float depth=mix(3.4,6.2,finish);
 vec3 origin=vec3((vGlPane.x-.5)*w,vGlPane.y*h,0.);
 vec3 direction=vec3(dot(ray.xz,normalize(vGlAxis)),ray.y,facing);
 // Full-height glazing starts near the floor; a narrower, shorter punched
 // opening normally has a raised sill. This also sets the visible furnishing height.
 float fullHeight=smoothstep(1.8,2.85,h);
 float sill=mix(.92,.15,fullHeight);
 vec3 lo=vec3(-max(3.2,w+mix(1.,2.,fullHeight))*.5,-sill,0.);
 vec3 hi=vec3(-lo.x,h+.6,depth);
 vec3 side=step(vec3(0.),direction);
 vec3 safeDir=mix(-max(abs(direction),vec3(.0001)),max(abs(direction),vec3(.0001)),side);
 vec3 hitT=(mix(lo,hi,side)-origin)/safeDir;
 float t=min(min(hitT.x,hitT.y),hitT.z);
 vec3 hit=origin+t*direction;
 vec3 wall=mix(vec3(.65,.58,.51),vec3(.65,.68,.69),finish);
 bool horizontal=hitT.y==t;
 bool ceiling=horizontal && direction.y>0.;
 vec3 base=horizontal ? (ceiling ? vec3(.84,.83,.80) : vec3(.31,.24,.18)) : wall;
 // A skirting board runs along the floor. Some rooms give onto a darker
 // opening at the rear; this stays several metres behind the actual facade.
 if(!horizontal){
  float aboveFloor=hit.y-lo.y, aa=max(fwidth(aboveFloor),.006);
  base*=.80+.20*smoothstep(.095-aa,.095+aa,aboveFloor);
  if(hitT.z==t && finish>.39){
   float offset=(glHash(seed+3.4)-.5)*(2.*hi.x-1.15);
   float door=max(abs(hit.x-offset)-.44,aboveFloor-2.05);
   float edgeWidth=max(fwidth(door),.006);
   float opening=1.-smoothstep(-edgeWidth,edgeWidth,door);
   base*=1.-.79*opening;
  }
 }
 // Soft contact shade at the floor and side returns gives the moving eye real
 // parallax cues even in unoccupied rooms. Use distances in metres.
 vec3 edge=min(hit-lo,hi-hit);
 float corner=horizontal ? min(edge.x,edge.z) : (hitT.x==t ? min(edge.y,edge.z) : min(edge.x,edge.y));
 base*=mix(.66,1.,smoothstep(0.,.45,corner));

 vec3 surfaceN=horizontal ? vec3(0.,direction.y>0.?-1.:1.,0.) :
    (hitT.x==t ? vec3(direction.x>0.?-1.:1.,0.,0.) : vec3(0.,0.,-1.));
 float layoutChoice=glHash(seed+8.3);
 // Pieces sit 0.3–1.6 m behind the pane. Only broad/tall glass gets the
 // continuous worktop and shelves; smaller rooms have a lower side cabinet.
 float front=mix(.32,.76,finish), handed=glHash(seed+5.1)>.5?1.:-1.;
 float center=handed*min(.29*w,1.15), sizeX=min(hi.x*.73,mix(.67,1.35,fullHeight));
 float top=lo.y+mix(.89,1.08,fullHeight), back=front+mix(.46,.76,finish);
 float roomT=t;
 vec3 inv=1./safeDir;
 if(layoutChoice>.23){
  vec3 wood=mix(vec3(.23,.15,.091),vec3(.16,.20,.21),finish);
  glSolid(origin,inv,vec3(center-sizeX+.09,lo.y+.07,front+.10),
    vec3(center+sizeX-.09,top-.055,back-.035),wood,t,surfaceN,base);
  glSolid(origin,inv,vec3(center-sizeX,top-.055,front),
    vec3(center+sizeX,top,back),wood*1.48,t,surfaceN,base);
  if(fullHeight>.22 && layoutChoice>.47){
   float side=handed*(hi.x-.17);
   vec3 shelf=vec3(.38,.32,.25);
   glSolid(origin,inv,vec3(side-.17,top+.44,front+.17),
     vec3(side+.17,top+.49,min(back+1.04,hi.z)),shelf,t,surfaceN,base);
   glSolid(origin,inv,vec3(side-.17,top+.90,front+.17),
     vec3(side+.17,top+.95,min(back+1.04,hi.z)),shelf,t,surfaceN,base);
  }
 }
 if(t<roomT){hit=origin+t*direction;ceiling=false;}
 // At oblique angles a ray meets a side wall close to the glass. A recessed
 // return breaks up its flat face even when no far wall is visible.
 else if(!horizontal && hitT.x==roomT){
  float trim=abs(hit.z-(.44+.43*finish));
  float aa=max(fwidth(trim),.012);
  base*=.74+.26*smoothstep(.032-aa,.032+aa,trim);
 }
 float daylight=.04+.15*exp(-.43*hit.z);
 if(uGlNight>.001 && mod(vGlId,256.)<123.){
  // One stable lighting state across an opening, including divided panes.
  // A recessed ceiling source lights the wall it faces and is visible only
  // along rays that actually intersect the ceiling near that source.
  float lampOffset=glHash(seed+2.9), color=mod(floor(vGlId/256.),256.)/255.;
  vec3 source=vec3((lampOffset-.5)*min(2.2,hi.x),hi.y-.16,depth*(.42+.19*finish));
  vec3 delta=source-hit;
  float r2=max(dot(delta,delta),.08);
  float diffuse=max(dot(surfaceN,delta*inversesqrt(r2)),0.);
  float direct=3.6*diffuse/(1.+.48*r2);
  vec3 lampColor=mix(vec3(1.,.49,.23),vec3(1.,.83,.64),color);
  float brightness=mix(.47,1.1,floor(vGlId/65536.)/255.);
  float aperture=0.;
  if(ceiling){
   float r=length(hit.xz-source.xz);
   float aa=clamp(fwidth(r),.012,.22);
   aperture=1.-smoothstep(.13-aa,.13+aa,r);
   if(fullHeight>.4 && layoutChoice>.42){
    // A longer source suspended just inside broad glazing reads from the road.
    vec2 line=abs(hit.xz-vec2(.14*handed,.72+.42*finish))-vec2(min(w*.29,.90),.046);
    float rim=max(line.x,line.y);
    float width=max(fwidth(rim),.012);
    aperture=max(aperture,1.-smoothstep(-width,width,rim));
   }
  }
  lamp=uGlNight*lampColor*brightness*(base*(.14+direct)+vec3(aperture*2.8));
 }
 vec3 reflectedDay=base*daylight;
 // Partially lowered fabric shades sit behind some narrow openings. Their
 // hems shift against the deeper room when the driver passes. Choose them
 // only for dark rooms so the recorded emitting aperture stays unchanged.
 if(mod(vGlId,256.)>=171. && w<2.6 && h<2.9){
  vec3 blind=origin+direction*(.095/direction.z);
  float bottom=mix(.38,.83,finish)*h;
  float edge=max(bottom-blind.y,abs(blind.x)-(.5*w+.04));
  float aa=max(fwidth(edge),.008);
  float coverage=1.-smoothstep(-aa,aa,edge);
  vec3 cloth=mix(vec3(.32,.28,.22),vec3(.56,.57,.56),finish);
  float hem=1.-smoothstep(.009,.018+aa,abs(blind.y-bottom-.012));
  reflectedDay=mix(reflectedDay,cloth*(.57-.09*hem),coverage);
 }
 return reflectedDay;
}
`;

const GLASS = /^(glass|glass_dark|glass_light|glass_blue|glass_clear)$/;
export function isGlazing(cls: string) { return GLASS.test(cls); }
// how much light the glass lets through, by class (tinted glazing shows what is behind it darker)
const TRANSMISSION: Record<string, number> = {glass: .72, glass_dark: .42, glass_light: .8, glass_blue: .6, glass_clear: .85};

/** One window of ordinary size: its horizontal axis (x, z), its extent along it and up, and its glass plane's offset
 *  across the axis (u = dot(P.xz, axis), off = dot(P.xz, (-az, ax))). */
export type GlazedWindow = {ax: number; az: number; u0: number; u1: number; y0: number; y1: number; off: number};

/** Optional inputs to the night spill bake. For a closed or double-sided pane
 * the direction along its normal must be resolved against building occupancy. */
export type GlazingEmitter = {
  center: [number, number, number]; normal: [number, number]; normalKnown: boolean;
  width: number; height: number; intensity: number; color: [number, number, number];
};

// One exactly representable 24-bit integer carries independent occupancy,
// temperature and brightness bytes from the mesh into the room shader.
const roomCode = (hash: number) => Math.floor(hash * (16777216 / 1000));

/** Gives every pane of the given glazing meshes its own coordinates across its face (glPane: u, v, width, height) and an
 *  identity (glId). A pane is a connected group of triangles (the generator draws each as a thin box). Returns the
 *  windows of ordinary size found (for the walls around them: weathering.ts, facademap.ts). */
function* paneAttributeSteps(meshes: THREE.Mesh[], emitters?: GlazingEmitter[], classes?: Map<THREE.Material, string>):Generator<void,GlazedWindow[]> {
  const P = new THREE.Vector3(); const windows: GlazedWindow[] = [];
  for (const mesh of meshes) {
    yield;
    const g = mesh.geometry; if (g.attributes.glPane) continue;
    mesh.updateWorldMatrix(true, false);
    const pos = g.attributes.position, n = pos.count, idx = g.index;
    const world = new Float32Array(n * 3);
    for (let i = 0; i < n; i++) { if(i%1024===0)yield; P.fromBufferAttribute(pos, i).applyMatrix4(mesh.matrixWorld); world[3 * i] = P.x; world[3 * i + 1] = P.y; world[3 * i + 2] = P.z; }
    // vertices at one position (1 mm) are one; triangles join their vertices
    const key = new Map<string, number>(), rep = new Int32Array(n);
    for (let i = 0; i < n; i++) { if(i%1024===0)yield; const k = Math.round(world[3 * i] * 1000) + ',' + Math.round(world[3 * i + 1] * 1000) + ',' + Math.round(world[3 * i + 2] * 1000); let r = key.get(k); if (r === undefined) { r = i; key.set(k, i); } rep[i] = r; }
    const parent = new Int32Array(n); for (let i = 0; i < n; i++) parent[i] = i;
    const find = (x: number): number => { while (parent[x] !== x) { parent[x] = parent[parent[x]]; x = parent[x]; } return x; };
    const join = (a: number, b: number) => { a = find(a); b = find(b); if (a !== b) parent[b] = a; };
    const tris = idx ? idx.count : n;
    for (let t = 0; t < tris; t += 3) { if(t%3072===0)yield; const a = rep[idx ? idx.getX(t) : t], b = rep[idx ? idx.getX(t + 1) : t + 1], c = rep[idx ? idx.getX(t + 2) : t + 2]; join(a, b); join(a, c); }
    const comps = new Map<number, number[]>();
    for (let i = 0; i < n; i++) { if(i%1024===0)yield; const r = find(rep[i]); let c = comps.get(r); if (!c) { c = []; comps.set(r, c); } c.push(i); }
    const material = Array.isArray(mesh.material) ? undefined : mesh.material;
    const canEmit = !!emitters && material instanceof THREE.MeshStandardMaterial && !material.map;
    const transmission = canEmit ? TRANSMISSION[classes?.get(material) ?? material.name] ?? .72 : 0;
    const directions = new Map<number, {x: number; z: number; area: number}>();
    if (canEmit && material.side === THREE.FrontSide) {
      for (let t = 0; t < tris; t += 3) { if(t%3072===0)yield;
        const ai = idx ? idx.getX(t) : t, a = ai*3, b = (idx ? idx.getX(t+1) : t+1)*3, c = (idx ? idx.getX(t+2) : t+2)*3;
        const dx = world[b]-world[a], dy = world[b+1]-world[a+1], dz = world[b+2]-world[a+2];
        const ey = world[c+1]-world[a+1], x = dy*(world[c+2]-world[a+2])-dz*ey, z = dx*ey-dy*(world[c]-world[a]);
        const root = find(rep[ai]), normal = directions.get(root) ?? {x: 0, z: 0, area: 0};
        normal.x += x; normal.z += z; normal.area += Math.hypot(x,z); directions.set(root, normal);
      }
    }
    // each pane: its horizontal axis (the longest horizontal extent between its vertices), its extents along it and up,
    // and its plane's offset across it
    type Pane = {root: number; verts: number[]; ax: number; az: number; umin: number; umax: number; ymin: number; ymax: number; off: number; cx: number; cy: number; cz: number};
    const panes: Pane[] = [];
    for (const [root, verts] of comps) {
      yield;
      let ax = 1, az = 0, best = -1; const i0 = verts[0];
      for (const i of verts) { if(i%1024===0)yield; const dx = world[3 * i] - world[3 * i0], dz = world[3 * i + 2] - world[3 * i0 + 2], d = dx * dx + dz * dz; if (d > best) { best = d; ax = dx; az = dz; } }
      const l = Math.hypot(ax, az) || 1; ax /= l; az /= l; if (ax < 0 || (ax === 0 && az < 0)) { ax = -ax; az = -az; }
      let umin = Infinity, umax = -Infinity, ymin = Infinity, ymax = -Infinity, cx = 0, cy = 0, cz = 0;
      for (const i of verts) { if(i%1024===0)yield; const u = world[3 * i] * ax + world[3 * i + 2] * az; umin = Math.min(umin, u); umax = Math.max(umax, u); ymin = Math.min(ymin, world[3 * i + 1]); ymax = Math.max(ymax, world[3 * i + 1]); cx += world[3 * i]; cy += world[3 * i + 1]; cz += world[3 * i + 2]; }
      const m = verts.length; cx /= m; cy /= m; cz /= m;
      panes.push({root, verts, ax, az, umin, umax, ymin, ymax, off: -cx * az + cz * ax, cx, cy, cz});
    }
    // windows: panes in one plane that touch across a mullion (gaps up to 15 cm) are one window
    const wp = new Int32Array(panes.length); for (let i = 0; i < panes.length; i++) wp[i] = i;
    const wf = (x: number): number => { while (wp[x] !== x) { wp[x] = wp[wp[x]]; x = wp[x]; } return x; };
    const grid = new Map<string, number[]>(), cellOf = (x: number, z: number) => Math.floor(x / 3) + ',' + Math.floor(z / 3);
    for(const [i,p] of panes.entries()){yield;const k=cellOf(p.cx,p.cz);const c=grid.get(k);if(c)c.push(i);else grid.set(k,[i]);}
    for(const [i,p] of panes.entries()){
      yield;
      const gx = Math.floor(p.cx / 3), gz = Math.floor(p.cz / 3);
      for (let dx = -1; dx <= 1; dx++) for (let dz = -1; dz <= 1; dz++) for (const j of grid.get((gx + dx) + ',' + (gz + dz)) ?? []) {
        if (j <= i) continue; const q = panes[j];
        if (Math.abs(p.ax * q.ax + p.az * q.az) < .985 || Math.abs(p.off - (-q.cx * p.az + q.cz * p.ax)) > .25) continue;
        const qc = q.cx * p.ax + q.cz * p.az, qu0 = qc - (q.umax - q.umin) / 2, qu1 = qc + (q.umax - q.umin) / 2;   // q's extent along p's axis
        if (q.ymin > p.ymax + .15 || p.ymin > q.ymax + .15 || qu0 > p.umax + .15 || p.umin > qu1 + .15) continue;
        const a = wf(i), b = wf(j); if (a !== b) wp[b] = a;
      }
    }
    const groups=new Map<number,number[]>();for(let i=0;i<panes.length;i++){if(i%1024===0)yield;const r=wf(i),g0=groups.get(r);if(g0)g0.push(i);else groups.set(r,[i]);}
    const pane = new Float32Array(n * 4), ident = new Float32Array(n), axes = new Float32Array(n * 2);
    const hash = (a: number, b: number, c: number) => (Math.abs(Math.sin(a * 12.9898 + b * 78.233 + c * 37.719) * 43758.5453)) % 1 * 1000;
    for (const [, members] of groups) {
      yield;
      const p0 = panes[members[0]]; let umin = Infinity, umax = -Infinity, ymin = Infinity, ymax = -Infinity, cx = 0, cy = 0, cz = 0;
      for (const k of members) { const p = panes[k]; for (const i of p.verts) { if(i%1024===0)yield; const u = world[3 * i] * p0.ax + world[3 * i + 2] * p0.az; umin = Math.min(umin, u); umax = Math.max(umax, u); ymin = Math.min(ymin, world[3 * i + 1]); ymax = Math.max(ymax, world[3 * i + 1]); } cx += p.cx; cy += p.cy; cz += p.cz; }
      const wid = hash(cx / members.length, cy / members.length, cz / members.length);
      // a window of ordinary size takes one treatment across all its panes; a curtain wall (one great connected glazing)
      // takes one per room-sized cell (4 m bays, 3.6 m storeys), each pane across its own face
      const ordinary = umax - umin <= 4.5 && ymax - ymin <= 4.2;
      if (ordinary && umax - umin > .2 && ymax - ymin > .3) {
        windows.push({ax: p0.ax, az: p0.az, u0: umin, u1: umax, y0: ymin, y1: ymax, off: p0.off});
        const code = roomCode(wid);
        if (emitters && canEmit && (code & 255) < 123) {
          const u = (umin + umax)*.5, nx = -p0.az, nz = p0.ax;
          let signed = 0, area = 0;
          for (const k of members) {
            const n = directions.get(panes[k].root);
            if (n) { signed += n.x*nx+n.z*nz; area += n.area; }
          }
          const known = area > 0 && Math.abs(signed) > area*.8, sense = known && signed < 0 ? -1 : 1;
          const warmth = ((code >>> 8) & 255) / 255;
          emitters.push({center:[p0.ax*u+nx*p0.off,(ymax+ymin)*.5,p0.az*u+nz*p0.off],
            normal:[sense*nx,sense*nz],normalKnown:known,width:umax-umin,height:ymax-ymin,
            intensity:transmission*.92*(.47+.63*(code >>> 16)/255),color:[1,.49+.34*warmth,.23+.41*warmth]});
        }
      }
      for (const k of members) {
        const p = panes[k];
        let pu0 = umin, pu1 = umax;
        if (!ordinary) {
          pu0 = Infinity; pu1 = -Infinity;
          for (const i of p.verts) { if(i%1024===0)yield;
            const u = world[3*i]*p0.ax + world[3*i+2]*p0.az;
            pu0 = Math.min(pu0, u); pu1 = Math.max(pu1, u);
          }
        }
        const py0 = ordinary ? ymin : p.ymin, py1 = ordinary ? ymax : p.ymax, w = Math.max(pu1 - pu0, 1e-3), h = Math.max(py1 - py0, 1e-3);
        const id = roomCode(ordinary ? wid : hash(wid, Math.floor(p.ymin / 3.6), Math.floor((p.cx * p0.ax + p.cz * p0.az) / 4)));
        for (const i of p.verts) { if(i%1024===0)yield; const u = world[3 * i] * p0.ax + world[3 * i + 2] * p0.az; pane[4 * i] = (u - pu0) / w; pane[4 * i + 1] = (world[3 * i + 1] - py0) / h; pane[4 * i + 2] = w; pane[4 * i + 3] = h; ident[i] = id; axes[2 * i] = p0.ax; axes[2 * i + 1] = p0.az; }
      }
    }
    g.setAttribute('glPane', new THREE.BufferAttribute(pane, 4)); g.setAttribute('glId', new THREE.BufferAttribute(ident, 1)); g.setAttribute('glAxis', new THREE.BufferAttribute(axes, 2));
  }
  return windows;
}

export function paneAttributes(meshes:THREE.Mesh[],emitters?:GlazingEmitter[],classes?:Map<THREE.Material,string>) {
 return finishStaticSteps(paneAttributeSteps(meshes,emitters,classes));
}
export function paneAttributesAsync(meshes:THREE.Mesh[],emitters:GlazingEmitter[]|undefined,classes:Map<THREE.Material,string>|undefined,yieldCpu:()=>Promise<void>) {
 return finishStaticStepsAsync(paneAttributeSteps(meshes,emitters,classes),yieldCpu);
}

/** Draws what stands behind each given glazing material's panes (chained after any existing onBeforeCompile). `cls` names
 *  each material's class, for its transmission. The meshes must carry paneAttributes(). */
export function glazeMaterials(materials: Iterable<THREE.MeshStandardMaterial>, cls?: Map<THREE.Material, string>, night: THREE.IUniform<number> = {value:0}) {
  for (const m of materials) {
    if (m.userData.glazed || m.map) continue; m.userData.glazed = true;
    const T = TRANSMISSION[cls?.get(m) ?? 'glass'] ?? .72;
    const previous = m.onBeforeCompile.bind(m);
    m.onBeforeCompile = (shader, renderer) => {
      previous(shader, renderer);
      shader.uniforms.uGlT = {value: T};
      shader.uniforms.uGlNight = night;
      shader.vertexShader = shader.vertexShader.replace('#include <common>', '#include <common>\nattribute vec4 glPane;attribute float glId;attribute vec2 glAxis;varying vec3 vGlPos;varying vec3 vGlNrm;varying vec4 vGlPane;flat varying float vGlId;varying vec2 vGlAxis;')
        .replace('#include <project_vertex>', '#include <project_vertex>\nvGlPos=(modelMatrix*vec4(transformed,1.)).xyz;vGlNrm=normalize(mat3(modelMatrix)*objectNormal);vGlPane=glPane;vGlId=glId;vGlAxis=glAxis;');
      shader.fragmentShader = shader.fragmentShader.replace('#include <common>', '#include <common>\n' + HEADER)
        .replace('#include <map_fragment>', '#include <map_fragment>\nvec3 glLamp;float glPass;vec3 glLayer=glRoom(glLamp,glPass);diffuseColor.rgb=glLayer*glPass;')
        .replace('#include <emissivemap_fragment>', '#include <emissivemap_fragment>\ntotalEmissiveRadiance+=glLamp*glPass;')
        .replace('#include <lights_physical_fragment>', THREE.ShaderChunk.lights_physical_fragment.replace('mix( vec3( 0.04 ), diffuseColor.rgb, metalnessFactor )', 'mix( vec3( 0.08 ), diffuseColor.rgb, metalnessFactor )'));
    };
    const key = m.customProgramCacheKey.bind(m);
    m.customProgramCacheKey = () => key() + '|glazed7-room-interiors';
    m.needsUpdate = true;
  }
}
