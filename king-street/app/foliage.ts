import * as THREE from 'three';
import {computeStaticBounds} from './static-detail-geometry.ts';
import {leafSampleHash} from './tree-foliage-lod.ts';

/** Finish the cylindrical limbs emitted by the tree skeleton before batching.
 * Each sector has one root cap, two side faces and one tip cap. Confirm the
 * shared geometry before touching it so other bark meshes pass through. */
export function taperTreeLimbs(mesh: THREE.Mesh) {
  const geometry = mesh.geometry;
  const positions = geometry.getAttribute('position');
  const normals = geometry.getAttribute('normal');
  const indices = geometry.index;
  if (!indices || geometry.userData.limbsFinished) return 0;
  mesh.updateWorldMatrix(true, false);
  const worldUp = new THREE.Vector3(0, 1, 0).transformDirection(mesh.matrixWorld.clone().invert());
  const get = (index: number, out: THREE.Vector3) => out.fromBufferAttribute(positions, index);
  const root = new THREE.Vector3(), tip = new THREE.Vector3(), axis = new THREE.Vector3();
  const vertex = new THREE.Vector3(), normal = new THREE.Vector3();
  const visited = new Uint8Array(positions.count);
  let changed = 0;
  for (let start = 0; start + 11 < indices.count;) {
    const rootIndex = indices.getX(start + 1), tipIndex = indices.getX(start + 11);
    let end = start + 12;
    while (end + 11 < indices.count && indices.getX(end + 1) === rootIndex && indices.getX(end + 11) === tipIndex) end += 12;
    const first = start;
    start = end;
    const sectors = (end - first) / 12;
    // Stream ownership may cut a few faces off a cylinder at the cell edge.
    // Recover at triangle boundaries, rather than rejecting every intact limb
    // in that cell because its total index count is no longer divisible by 12.
    if (sectors < 5 || sectors > 32) {start=first+3;continue;}
    get(rootIndex, root);
    get(tipIndex, tip);
    axis.subVectors(tip, root);
    const length = axis.length();
    if (!length) continue;
    axis.divideScalar(length);
    const upward = axis.dot(worldUp);
    if (upward < .12) continue;
    const radius = get(indices.getX(first), vertex).distanceTo(root);
    if (radius <= 0 || length < radius * 5) continue;
    // The two rings share a radius on a cylindrical limb. Preserve existing
    // tapers, trunks, palm collars, and naturally irregular branch meshes.
    let valid = true;
    for (let i = first; i < end && valid; i += 12) {
      const baseRadius = get(indices.getX(i), vertex).distanceTo(root);
      const tipRadius = get(indices.getX(i + 9), vertex).distanceTo(tip);
      valid = Math.abs(baseRadius - radius) < radius * .015 && Math.abs(tipRadius - radius) < radius * .015;
    }
    if (!valid) {start=first+3;continue;}
    const trunk=upward>.97;
    const endRadius = Math.min(radius, Math.max(.007, radius * (trunk ? .78 : .18)));
    const slope = (radius - endRadius) / length;
    for (let i = first; i < end; i++) {
      const index = indices.getX(i);
      if (visited[index]) continue;
      visited[index] = 1;
      get(index, vertex);
      if (vertex.sub(root).dot(axis) > length / 2) {
        get(index, vertex).sub(tip).multiplyScalar(endRadius / radius).add(tip);
        positions.setXYZ(index, vertex.x, vertex.y, vertex.z);
      }
      if (normals) {
        normal.fromBufferAttribute(normals, index);
        if (Math.abs(normal.dot(axis)) < .2) {
          normal.addScaledVector(axis, slope).normalize();
          normals.setXYZ(index, normal.x, normal.y, normal.z);
        }
      }
    }
    changed++;
  }
  geometry.userData.limbsFinished = true;
  if (changed) {
    positions.needsUpdate = true;
    if (normals) normals.needsUpdate = true;
  }
  return changed;
}

// Tree canopies. The generated trees carry a few hundred small flat leaf quads each, so a street tree read as bare
// sticks with green confetti. Each leaf quad becomes a leaf-cluster card here: a larger alpha-tested card showing
// a connected branch with overlapping leaves, turned at random around the quad's own position, so a crown keeps the
// shape and size the pipeline gave it but becomes a dense mass. Cards are shaded with normals pointing out from the
// local crown centre, so a canopy is lit as a rounded volume (sunlit side, shaded underside) instead of as scattered
// flat planes. Leaf colours stay the pipeline's three greens.

function rng(seed: number) { return () => { seed |= 0; seed = seed + 0x6d2b79f5 | 0; let t = Math.imul(seed ^ seed >>> 15, 1 | seed); t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t; return ((t ^ t >>> 14) >>> 0) / 4294967296; }; }

// The branch texture stores relative colour and cutout coverage. Per-tree
// vertex colours supply the canopy tint; every card shares one mip chain.
export function loadLeafTexture() {
  const texture = new THREE.TextureLoader().load('/foliage/broadleaf-spray.webp');
  texture.colorSpace = THREE.SRGBColorSpace;
  texture.anisotropy = 4;
  return texture;
}

// A full-city package has about one million leaf cards. An 80m cell submitted
// tens of thousands of leaves whenever just one crown touched a view edge.
// Smaller cells retain every card, while beauty/probe/cascade frusta reject
// offscreen crowns before their alpha-tested triangles reach the GPU.
export const FOLIAGE_CELL_M = 32;

/** Build identical seeded cards with independently culled spatial bounds. */
export function buildFoliage(leafMeshes: THREE.Mesh[], texture:THREE.Texture, cellM = FOLIAGE_CELL_M,
  sizeRange:readonly [number,number]=[1.05,1.65]) {
  // All vertices of a compact fan mesh are card centres. Old packages may
  // contain two-triangle quads, four-triangle fans, or both in the same mesh.
  const capacity = leafMeshes.reduce((sum, mesh) => {
    const g = mesh.geometry;
    return sum + (mesh.userData.foliageFanCentres ? g.attributes.position.count :
      Math.ceil((g.index?.count ?? g.attributes.position.count) / 6));
  }, 0);
  const centroids = new Float64Array(capacity * 3);
  const sourceColors = new Float32Array(capacity * 3);
  const sourceScales = new Float32Array(capacity);
  const tintRuns: {end: number; color: THREE.Color}[] = [];
  let n = 0;
  for (const mesh of leafMeshes) {
    mesh.updateWorldMatrix(true, false);
    const g = mesh.geometry, p = g.attributes.position, idx = g.index;
    const v = new THREE.Vector3(), s = new THREE.Vector3();
    const push = (row:number) => {
      centroids[n * 3] = s.x;
      centroids[n * 3 + 1] = s.y;
      centroids[n * 3 + 2] = s.z;
      const color=g.getAttribute('color');
      sourceColors[n*3]=color?color.getX(row):1;
      sourceColors[n*3+1]=color?color.getY(row):1;
      sourceColors[n*3+2]=color?color.getZ(row):1;
      sourceScales[n]=g.getAttribute('foliageScale')?.getX(row)??1;
      n++;
    };
    if (mesh.userData.foliageFanCentres) {
      for (let i = 0; i < p.count; i++) {
        s.fromBufferAttribute(p, i).applyMatrix4(mesh.matrixWorld);
        push(i);
      }
    } else {
      const count = idx ? idx.count : p.count;
      const at = (i: number) => idx ? idx.getX(i) : i;
      const equal = (i: number, j: number) => p.getX(i) === p.getX(j) && p.getY(i) === p.getY(j) && p.getZ(i) === p.getZ(j);
      for (let i = 0; i + 2 < count;) {
        const k = at(i + 2);
        if (i + 11 < count && equal(k, at(i + 5)) && equal(k, at(i + 8)) && equal(k, at(i + 11))) {
          s.fromBufferAttribute(p, k).applyMatrix4(mesh.matrixWorld);
          i += 12;
        } else {
          const take = Math.min(count - i, 6);
          s.set(0, 0, 0);
          for (let j = 0; j < take; j++) s.add(v.fromBufferAttribute(p, at(i + j)).applyMatrix4(mesh.matrixWorld));
          s.multiplyScalar(1 / take);
          i += take;
        }
        push(k);
      }
    }
    tintRuns.push({end: n, color: (mesh.material as THREE.MeshStandardMaterial).color || new THREE.Color('#557c38')});
  }

  // Count each 1 m neighbourhood and spatial render cell in one pass. A card
  // keeps its cell number so the million-card geometry loop needs no hash
  // lookups or coordinate strings.
  type Cell = {id: number; x: number; z: number; sum: THREE.Vector3; count: number; centre: THREE.Vector3; bucket: number; top: number; roof: number; near: number};
  const columns = new Map<number, Map<number, Cell>>();
  const cells: Cell[] = [], counts: number[] = [], buckets = new Map<string, number>();
  const cellIds = new Uint32Array(n);
  for (let i = 0; i < n; i++) {
    const x = centroids[3 * i], y = centroids[3 * i + 1], z = centroids[3 * i + 2];
    const ix = Math.floor(x), iz = Math.floor(z);
    let column = columns.get(ix);
    if (!column) columns.set(ix, column = new Map());
    let cell = column.get(iz);
    if (!cell) {
      const key = Math.floor(ix / cellM) + ',' + Math.floor(iz / cellM);
      let bucket = buckets.get(key);
      if (bucket === undefined) {
        bucket = counts.length;
        counts.push(0);
        buckets.set(key, bucket);
      }
      cell = {id: cells.length, x: ix, z: iz, sum: new THREE.Vector3(), count: 0, centre: new THREE.Vector3(), bucket, top: -Infinity, roof: -Infinity, near: 0};
      cells.push(cell);
      column.set(iz, cell);
    }
    cell.sum.x += x;
    cell.sum.y += y;
    cell.sum.z += z;
    cell.count++;
    if (y > cell.top) cell.top = y;
    cellIds[i] = cell.id;
  }
  for (const cell of cells) {
    let count = 0;
    for (let dx = -2; dx <= 2; dx++) {
      const column = columns.get(cell.x + dx);
      if (!column) continue;
      for (let dz = -2; dz <= 2; dz++) {
        const neighbor = column.get(cell.z + dz);
        if (!neighbor) continue;
        cell.centre.add(neighbor.sum);
        count += neighbor.count;
        if (neighbor.top > cell.roof) cell.roof = neighbor.top;
      }
    }
    cell.centre.set(cell.centre.x / count, cell.centre.y / count, cell.centre.z / count);
    cell.near = count;
    counts[cell.bucket] += cell.count;
  }
  const chunks=counts.map(count=>({
    next:0,pos:new Float32Array(count*12),nrm:new Int16Array(count*12),
    card:new Int16Array(count*12),uv:new Uint8Array(count*8),col:new Uint16Array(count*12),
    index:count*4>65535?new Uint32Array(count*6):new Uint16Array(count*6)
  }));
  const corners=[[-1,-1,0,0],[1,-1,1,0],[1,1,1,1],[-1,1,0,1]];
  const lichen=new THREE.Color(),horizontal=new THREE.Vector3(1,0,0);
  const p = new THREE.Vector3(), radial = new THREE.Vector3(), d = new THREE.Vector3(), a = new THREE.Vector3(), b = new THREE.Vector3(), up = new THREE.Vector3(0, 1, 0), q = new THREE.Vector3(), tint = new THREE.Color();
  let run = 0;
  for (let i = 0; i < n; i++) {
    const cell = cells[cellIds[i]], c = cell.centre;
    const chunk=chunks[cell.bucket],local=chunk.next++;
    while (i >= tintRuns[run].end) run++;
    const {pos,nrm,card,uv,col,index}=chunk;
    p.set(centroids[3 * i], centroids[3 * i + 1], centroids[3 * i + 2]);
    const r=rng(leafSampleHash(p.x,p.y,p.z));
    radial.copy(p).sub(c); if (radial.lengthSq() < 1e-4) radial.set(r() - .5, .5, r() - .5); radial.normalize();
    d.set(r() * 2 - 1, r() * 2 - 1, r() * 2 - 1).normalize().multiplyScalar(.8).addScaledVector(radial, .6).normalize();   // card facing
    a.crossVectors(d, Math.abs(d.y) > .9 ? horizontal : up).normalize(); b.crossVectors(d, a).normalize();
    const size = (sizeRange[0] + r() * (sizeRange[1]-sizeRange[0]))*sourceScales[i], rot = r() * Math.PI * 2, ca = Math.cos(rot), sa = Math.sin(rot);
    q.copy(a).multiplyScalar(ca).addScaledVector(b, sa); b.copy(a).multiplyScalar(-sa).addScaledVector(b, ca); a.copy(q);
    // canopy albedo: a crown in sun is a dark, slightly olive green (about 0.05-0.1), not the bright lawn green the
    // pipeline's leaf colours times this texture gave (about 0.15-0.28, a toy tree)
    tint.copy(tintRuns[run].color);tint.r*=sourceColors[i*3];tint.g*=sourceColors[i*3+1];tint.b*=sourceColors[i*3+2];
    tint.multiplyScalar(.76 + r() * .24); { const l = .2126 * tint.r + .7152 * tint.g + .0722 * tint.b; tint.lerp(lichen.setRGB(l * 1.08, l, l * .78), .18); }
    // Crown exposure, carried as the length of the card normal: 1 on the crown's top and sides, falling towards
    // .2 for a card a few metres under the local canopy top in the middle of the mass. The shader darkens sky and
    // sun light with it, so a crown has a dark interior and underside instead of one even green.
    const under = Math.min(1, Math.max(0, (cell.roof - p.y) / 2.4)), side = Math.min(1, Math.hypot(p.x - c.x, p.z - c.z) / 1.6);
    const exposure = cell.near < 6 ? 1 : 1 - .8 * under * (1 - .75 * side);
    for (let k = 0; k < 4; k++) {
      const [sx, sy, u, w] = corners[k]; const o = (4 * local + k) * 3;
      pos[o] = p.x + (a.x * sx + b.x * sy) * size / 2 + radial.x * .15; pos[o + 1] = p.y + (a.y * sx + b.y * sy) * size / 2 + radial.y * .15; pos[o + 2] = p.z + (a.z * sx + b.z * sy) * size / 2 + radial.z * .15;
      // shading normal: out from the crown centre, tipped up (a canopy is sky-lit from above)
      q.set(pos[o] - c.x, pos[o + 1] - c.y, pos[o + 2] - c.z).normalize().addScaledVector(up, .35).normalize();
      nrm[o]=Math.round(q.x*32767);nrm[o+1]=Math.round(q.y*32767);nrm[o+2]=Math.round(q.z*32767);
      col[o]=Math.round(tint.r*65535);col[o+1]=Math.round(tint.g*65535);col[o+2]=Math.round(tint.b*65535);
      card[o]=Math.round(d.x*exposure*32767);card[o+1]=Math.round(d.y*exposure*32767);card[o+2]=Math.round(d.z*exposure*32767);
      uv[(4 * local + k) * 2] = u; uv[(4 * local + k) * 2 + 1] = w;
    }
    const vertex=4*local,j=6*local;
    index[j]=vertex;index[j+1]=vertex+1;index[j+2]=vertex+2;
    index[j+3]=vertex;index[j+4]=vertex+2;index[j+5]=vertex+3;
  }
  const material=new THREE.MeshStandardMaterial({map:texture,vertexColors:true,alphaTest:.45,side:THREE.DoubleSide,roughness:.85,metalness:0});
  material.alphaToCoverage=true;material.name='foliage';
  const meshes=chunks.map(({pos,nrm,card,uv,col,index})=>{
    const g=new THREE.BufferGeometry();
    g.setAttribute('position',new THREE.BufferAttribute(pos,3));g.setAttribute('normal',new THREE.BufferAttribute(nrm,3,true));
    g.setAttribute('cardNormal',new THREE.BufferAttribute(card,3,true));g.setAttribute('uv',new THREE.BufferAttribute(uv,2));
    g.setAttribute('color',new THREE.BufferAttribute(col,3,true));g.setIndex(new THREE.BufferAttribute(index,1));
    computeStaticBounds(g);
    const mesh=new THREE.Mesh(g,material);mesh.name='vegetation___foliage';mesh.castShadow=mesh.receiveShadow=true;
    mesh.userData.layer='vegetation';mesh.userData.original=material;
    return mesh;
  });
  return {meshes,material};
}

/** Leaf cards turned edge-on to the view fade out (they smeared into streaks), and the leaf alpha is sharpened with the
 *  texture's mip level so distant crowns stay crisp leaves instead of blurred blotches. Chained after the shadow
 *  cascades' onBeforeCompile, so call it after they have set the material up. */
/** Seconds, advanced by the viewer each frame: leaf cards sway and flutter in the vertex shader. */
export const foliageWind = {value: 0};
export function finishFoliage(material: THREE.MeshStandardMaterial) {
  // The leaf's transmitted light also needs a sun shadow on its back face.
  const previous = material.onBeforeCompile.bind(material);
  material.onBeforeCompile = (shader, renderer) => {
    previous(shader, renderer);
    shader.uniforms.uFoliageTime = foliageWind;
    shader.vertexShader = shader.vertexShader.replace('#include <common>', '#include <common>\nattribute vec3 cardNormal;uniform float uFoliageTime;varying vec3 vCardN;varying vec3 vCardPos;varying float vCrown;')
      // Wind: the branch a card hangs on swings slowly with a gust that travels across the street, and the card itself
      // twists about its centre. Centimetres only, so the unmoved shadow-map copy still lines up.
      .replace('#include <begin_vertex>', `#include <begin_vertex>
 {vec3 wp=(modelMatrix*vec4(position,1.)).xyz;vec2 corner=vec2(uv)*2.-1.;float ph=dot(cardNormal,vec3(37.,17.,29.));
  float gust=sin(dot(wp.xz,vec2(.11,.07))+uFoliageTime*.7)+.5*sin(dot(wp.xz,vec2(-.05,.13))+uFoliageTime*1.3);
  transformed.xz+=vec2(.8,.6)*gust*.016;
  transformed+=normalize(cardNormal)*corner.x*sin(uFoliageTime*1.7+ph+wp.y)*.018*(.7+.3*gust);
  transformed.y+=corner.y*sin(uFoliageTime*1.3+ph*1.7)*.012;}`)
      .replace('#include <project_vertex>', '#include <project_vertex>\nvCrown=length(cardNormal);vCardN=normalize(mat3(modelMatrix)*cardNormal);vCardPos=(modelMatrix*vec4(transformed,1.)).xyz;');
    shader.fragmentShader = shader.fragmentShader.replace('#include <common>', '#define LEAF_BACKLIGHT 1\n#include <common>\nvarying vec3 vCardN;varying vec3 vCardPos;varying float vCrown;')
      // A thin sunlit leaf transmits some daylight towards the camera. The
      // incident colour already includes the active cascade's shadow and fade.
      .replace('#include <lights_physical_pars_fragment>', `#include <lights_physical_pars_fragment>
void leafDirect(const in IncidentLight light, const in vec3 p, const in vec3 n,
  const in vec3 eye, const in vec3 coatN, const in PhysicalMaterial m, inout ReflectedLight r) {
  // Cards deep in the crown sit behind more leaves than the shadow map's one sample resolves.
  IncidentLight sun=light;sun.color*=mix(.4,1.,vCrown);
  RE_Direct_Physical(sun,p,n,eye,coatN,m,r);
  float back=clamp(.5-.5*dot(n,light.direction),0.,1.);
  float forwardScatter=pow(max(dot(eye,-light.direction),0.),3.);
  vec3 transmittance=m.diffuseColor*vec3(1.25,1.9,.36);
  r.directDiffuse+=sun.color*transmittance*RECIPROCAL_PI*back*(.2+1.6*forwardScatter)*vCrown;
}
#undef RE_Direct
#define RE_Direct leafDirect`)
      // Crown normals give diffuse depth, but their far side cannot form a
      // mirror facing the eye. Keep the smaller sheen of a collection of leaves.
      .replace('#include <lights_fragment_end>','#include <lights_fragment_end>\nreflectedLight.directSpecular *= .25*smoothstep(.01,.30,dot(normal,geometryViewDir));reflectedLight.indirectDiffuse*=mix(.28,1.,vCrown);')
      // Lighting follows the crown's outward normal on either side of a leaf card.
      .replace('#include <normal_fragment_begin>', THREE.ShaderChunk.normal_fragment_begin.replace('normal *= faceDirection;', ''))
      .replace('#include <common>', `#include <common>
float lfHash(vec2 p){return fract(sin(dot(p,vec2(127.1,311.7)))*43758.5453);}
float lfNoise(vec2 p){vec2 i=floor(p),f=fract(p);f=f*f*(3.-2.*f);
 return mix(mix(lfHash(i),lfHash(i+vec2(1.,0.)),f.x),mix(lfHash(i+vec2(0.,1.)),lfHash(i+vec2(1.,1.)),f.x),f.y);}`)
      .replace('#include <map_fragment>', `#include <map_fragment>
 {vec2 g=vMapUv*512.;float lod=max(0.,log2(max(length(dFdx(g)),length(dFdy(g)))));diffuseColor.a=min(1.,diffuseColor.a*(1.+min(lod,5.)*.17));
  // Outer leaves are lighter and yellower than the ones inside the crown.
  diffuseColor.rgb*=mix(vec3(.66,.76,.86),vec3(1.2,1.17,.9),vCrown);
  // Once the texture's own leaves are below a pixel, a card is one smooth blob shaped like a giant leaf. Clumps of
  // lighter and darker leaves, with gaps between them, keep it reading as a mass of foliage.
  float far=smoothstep(1.2,3.2,log2(max(1.,min(length(dFdx(g)),length(dFdy(g))))));
  if(far>0.){vec2 q=vMapUv*7.+dot(vCardN,vec3(12.9,78.2,37.7));float clump=lfNoise(q)*.65+lfNoise(q*2.3+9.)*.35;
   diffuseColor.a*=mix(1.,smoothstep(.2,.42,clump)*1.3,far*.85);diffuseColor.rgb*=mix(1.,.62+.8*clump,far);}
  float edge=abs(dot(normalize(cameraPosition-vCardPos),normalize(vCardN)));diffuseColor.a*=smoothstep(.035,.25,edge);}`);
  };
  const key = material.customProgramCacheKey.bind(material);
  material.customProgramCacheKey = () => key() + '|foliage-thin5';
  material.needsUpdate = true;
}

// Bark: the generated trunks are one flat tan. Real bark is grey-brown, streaked up the trunk, patched with grey-green
// lichen. A 3D pattern stretched vertically gives any upright trunk vertical streaks without knowing where its axis is;
// they carry a little relief. Generic, not a species (palms and broadleaf trees share the class).
const BARK = /* glsl */`
varying vec3 vBkPos;
float bkHash(vec3 p){return fract(sin(dot(p,vec3(127.1,311.7,74.7)))*43758.5453);}
float bkNoise(vec3 p){vec3 i=floor(p),f=fract(p);f=f*f*(3.-2.*f);
 return mix(mix(mix(bkHash(i),bkHash(i+vec3(1,0,0)),f.x),mix(bkHash(i+vec3(0,1,0)),bkHash(i+vec3(1,1,0)),f.x),f.y),
            mix(mix(bkHash(i+vec3(0,0,1)),bkHash(i+vec3(1,0,1)),f.x),mix(bkHash(i+vec3(0,1,1)),bkHash(i+vec3(1,1,1)),f.x),f.y),f.z);}
float bkH;
vec3 bkBark(){
 // grey-brown, streaked up the trunk: noise stretched about twelvefold vertically, three octaves, low contrast; a little
 // lichen. Neutral on purpose: palms and broadleaf trunks share this class
 vec3 q=vBkPos*vec3(30.,2.5,30.);
 float keep=1.-smoothstep(.25,.5,max(fwidth(q.x),fwidth(q.z)));
 float st=bkNoise(q)*.5+bkNoise(q*vec3(2.3,2.1,2.3)+5.)*.3+bkNoise(q*vec3(5.,4.,5.)+11.)*.2;
 vec3 c=mix(vec3(.15,.13,.11),vec3(.23,.21,.18),bkNoise(vBkPos*vec3(4.,.8,4.)));
 c*=mix(1.,.72+.5*st,keep);
 float lichen=smoothstep(.65,.82,bkNoise(vBkPos*vec3(2.5,1.2,2.5)+7.));c=mix(c,vec3(.25,.27,.2),lichen*.45);
 bkH=mix(.5,st,keep);
 return c;
}
vec3 bkBump(vec3 n,float h,float k){vec3 dpx=dFdx(-vViewPosition),dpy=dFdy(-vViewPosition);float hx=dFdx(h),hy=dFdy(h);
 vec3 r1=cross(dpy,n),r2=cross(n,dpx);float det=dot(dpx,r1);vec3 g=sign(det)*(hx*r1+hy*r2);return normalize(abs(det)*n-k*g);}
`;

/** Fissured bark on the trunk materials (chained after the shadow cascades' onBeforeCompile). */
export function barkMaterials(materials: Iterable<THREE.MeshStandardMaterial>) {
  for (const m of materials) {
    if (m.userData.bark) continue; m.userData.bark = true; m.roughness = .92;
    const previous = m.onBeforeCompile.bind(m);
    m.onBeforeCompile = (shader, renderer) => {
      previous(shader, renderer);
      shader.vertexShader = shader.vertexShader.replace('#include <common>', '#include <common>\nvarying vec3 vBkPos;')
        .replace('#include <project_vertex>', '#include <project_vertex>\nvBkPos=(modelMatrix*vec4(transformed,1.)).xyz;');
      shader.fragmentShader = shader.fragmentShader.replace('#include <common>', '#include <common>\n' + BARK)
        .replace('#include <map_fragment>', '#include <map_fragment>\ndiffuseColor.rgb=bkBark();')
        .replace('#include <normal_fragment_maps>', '#include <normal_fragment_maps>\nnormal=bkBump(normal,bkH,.006);');
    };
    const key = m.customProgramCacheKey.bind(m); m.customProgramCacheKey = () => key() + '|bark'; m.needsUpdate = true;
  }
}
