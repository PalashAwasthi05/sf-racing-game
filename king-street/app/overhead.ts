import * as THREE from 'three';
import {mergeGeometries} from 'three/addons/utils/BufferGeometryUtils.js';

// Overhead street equipment: steel poles at the kerbs, cobra-head street lamps on arms, and the span and contact wires
// of trolleybus lines. Where they stand comes with the world (areas/overhead.py: OpenStreetMap's trolley_wire and lit
// streets, poles behind the kerbs of the built road surface). Keep one draw per material, with individual
// pole and wire bounds for street, shadow and reflection cameras on renderers with native multi-draw.

type Pole = {x: number; y: number; z: number; h: number; arm: [number, number] | null};
type Wire = {r: number; p: [number, number, number][]};
export type Overhead = {group: THREE.Group; materials: THREE.Material[]; circles: number[]; poles: number; wires: number; lamps: THREE.Vector4[]; lens: THREE.MeshStandardMaterial};

const ARM = 2.05, ARM_RISE = .35;

/** The overhead equipment of a world, or null when it has none. Coordinates are the world's (x east, y north, z up). */
export async function loadOverhead(url: string, multiDraw = false): Promise<Overhead | null> {
  const data = await fetch(url, {cache: 'no-cache'}).then(r => r.ok ? r.json() as Promise<{poles: Pole[]; wires: Wire[]}> : null).catch(() => null);
  if (!data || (!data.poles?.length && !data.wires?.length)) return null;
  const steel = new THREE.MeshStandardMaterial({color: '#3d4441', metalness: .45, roughness: .55});
  const concrete = new THREE.MeshStandardMaterial({color: '#8d8a83', roughness: .9});
  const head = new THREE.MeshStandardMaterial({color: '#7b7f80', metalness: .35, roughness: .45});
  const lens = new THREE.MeshStandardMaterial({color: '#e9e4d6', emissive: new THREE.Color('#fff1cf'), emissiveIntensity: .35, roughness: .3});
  const wire = new THREE.MeshStandardMaterial({color: '#18191b', metalness: .3, roughness: .5});
  const group = new THREE.Group(); group.name = 'overhead'; group.userData.layer = 'street_objects';
  const poles = data.poles ?? [], armed = poles.filter(p => p.arm);
  const shaft = new THREE.CylinderGeometry(.075, .12, 1, 10); shaft.translate(0, .5, 0);
  const footing = new THREE.CylinderGeometry(.22, .26, .45, 10); footing.translate(0, .1, 0);
  // the arm: a slim tube rising toward the road, along +x from the pole
  const armGeo = new THREE.CylinderGeometry(.035, .045, 1, 6); armGeo.rotateZ(-Math.PI / 2); armGeo.translate(.5, 0, 0);
  const headGeo = new THREE.SphereGeometry(1, 14, 8); headGeo.scale(.4, .1, .17);
  const lensGeo = new THREE.SphereGeometry(1, 12, 6, 0, Math.PI * 2, Math.PI / 2, Math.PI / 2); lensGeo.scale(.28, .05, .12);
  const mk = (g: THREE.BufferGeometry, m: THREE.Material, n: number, name: string) => {
    let mesh:THREE.InstancedMesh | THREE.BatchedMesh;
    if(multiDraw && n>0){
      mesh = new THREE.BatchedMesh(n, g.attributes.position.count, g.index!.count, m);
      mesh.sortObjects=false;
      const geometry=mesh.addGeometry(g);
      for(let i=0;i<n;i++)mesh.addInstance(geometry);
      g.dispose();
    }else mesh=new THREE.InstancedMesh(g,m,n);
    mesh.name='overhead__'+name;mesh.castShadow=name!=='lens';mesh.receiveShadow=true;
    group.add(mesh);return mesh;
  };
  const shafts = mk(shaft, steel, poles.length, 'pole'), footings = mk(footing, concrete, poles.length, 'footing');
  const arms = mk(armGeo, steel, armed.length, 'arm'), heads = mk(headGeo, head, armed.length, 'head'), lenses = mk(lensGeo, lens, armed.length, 'lens');
  const m4 = new THREE.Matrix4(), q = new THREE.Quaternion(), s = new THREE.Vector3(), p = new THREE.Vector3(), up = new THREE.Vector3(0, 1, 0), dir = new THREE.Vector3();
  const circles: number[] = [], lamps: THREE.Vector4[] = []; let a = 0;
  poles.forEach((pole, i) => {
    p.set(pole.x, pole.z, -pole.y);
    shafts.setMatrixAt(i, m4.compose(p, q.identity(), s.set(1, pole.h, 1)));
    footings.setMatrixAt(i, m4.compose(p, q.identity(), s.set(1, 1, 1)));
    circles.push(pole.x, -pole.y, .25);
    if (!pole.arm) return;
    // the arm leaves the pole 0.4 m below its top and rises ARM_RISE over ARM toward the road
    dir.set(pole.arm[0] * ARM, ARM_RISE, -pole.arm[1] * ARM); const len = dir.length(); dir.normalize();
    const base = p.clone().setY(pole.z + pole.h - .4);
    q.setFromUnitVectors(new THREE.Vector3(1, 0, 0), dir); arms.setMatrixAt(a, m4.compose(base, q, s.set(len, 1, 1)));
    const yaw = Math.atan2(pole.arm[1], pole.arm[0]); q.setFromEuler(new THREE.Euler(0, yaw, -.08));
    const tip = base.clone().addScaledVector(dir, len).add(new THREE.Vector3(pole.arm[0] * .3, -.06, -pole.arm[1] * .3));
    heads.setMatrixAt(a, m4.compose(tip, q, s.set(1, 1, 1)));
    tip.y -= .07;
    lenses.setMatrixAt(a, m4.compose(tip, q, s.set(1, 1, 1))); a++;
    lamps.push(new THREE.Vector4(tip.x, tip.y, tip.z, tip.y - pole.z));
  });
  for (const im of [shafts, footings, arms, heads, lenses]) { if(im instanceof THREE.InstancedMesh)im.instanceMatrix.needsUpdate = true; im.computeBoundingSphere(); }
  // wires: thin tubes along their sampled points (contact wires every 2 m, span wires in three points)
  const tubes: THREE.BufferGeometry[] = [];
  for (const w of data.wires ?? []) {
    if (w.p.length < 2) continue;
    const curve = new THREE.CatmullRomCurve3(w.p.map(([x, y, z]) => new THREE.Vector3(x, z, -y)));
    const g = new THREE.TubeGeometry(curve, Math.max(2, w.p.length * (w.p.length < 4 ? 4 : 1)), w.r, 4, false); g.deleteAttribute('uv'); tubes.push(g);
  }
  if (tubes.length) {
    let wires:THREE.Mesh;
    if(multiDraw){
      const vertices=tubes.reduce((n,g)=>n+g.attributes.position.count,0);
      const indices=tubes.reduce((n,g)=>n+g.index!.count,0);
      const batch=new THREE.BatchedMesh(tubes.length,vertices,indices,wire);
      batch.sortObjects=false;
      for(const tube of tubes)batch.addInstance(batch.addGeometry(tube));
      wires=batch;
    }else wires=new THREE.Mesh(mergeGeometries(tubes),wire);
    wires.name='overhead__wires';wires.castShadow=false;wires.receiveShadow=false;
    group.add(wires);tubes.forEach(t=>t.dispose());
  }
  group.updateMatrixWorld(true); group.traverse(o => { o.matrixAutoUpdate = false; });
  return {group, materials: [steel, concrete, head, lens, wire], circles, lamps, lens, poles: poles.length, wires: data.wires?.length ?? 0};
}
