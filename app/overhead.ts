import * as THREE from 'three';
import {mergeGeometries} from 'three/addons/utils/BufferGeometryUtils.js';

// Overhead street equipment: steel poles at the kerbs, cobra-head street lamps on arms, and the span and contact wires
// of trolleybus lines. Where they stand comes with the world (areas/overhead.py: OpenStreetMap's trolley_wire and lit
// streets, poles behind the kerbs of the built road surface). Poles, footings, arms, lamp heads and lenses are instanced
// (five draws), every wire is one merged tube mesh (one draw). Nothing here is location specific.

type Pole = {x: number; y: number; z: number; h: number; arm: [number, number] | null};
type Wire = {r: number; p: [number, number, number][]};
export type Overhead = {group: THREE.Group; materials: THREE.Material[]; circles: number[]; poles: number; wires: number};

const ARM = 2.05, ARM_RISE = .35;

/** The overhead equipment of a world, or null when it has none. Coordinates are the world's (x east, y north, z up). */
export async function loadOverhead(url: string): Promise<Overhead | null> {
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
  const mk = (g: THREE.BufferGeometry, m: THREE.Material, n: number, name: string) => { const im = new THREE.InstancedMesh(g, m, n); im.name = 'overhead__' + name; im.castShadow = name !== 'lens'; im.receiveShadow = true; group.add(im); return im; };
  const shafts = mk(shaft, steel, poles.length, 'pole'), footings = mk(footing, concrete, poles.length, 'footing');
  const arms = mk(armGeo, steel, armed.length, 'arm'), heads = mk(headGeo, head, armed.length, 'head'), lenses = mk(lensGeo, lens, armed.length, 'lens');
  const m4 = new THREE.Matrix4(), q = new THREE.Quaternion(), s = new THREE.Vector3(), p = new THREE.Vector3(), up = new THREE.Vector3(0, 1, 0), dir = new THREE.Vector3();
  const circles: number[] = []; let a = 0;
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
    lenses.setMatrixAt(a, m4.compose(tip.clone().setY(tip.y - .07), q, s.set(1, 1, 1))); a++;
  });
  for (const im of [shafts, footings, arms, heads, lenses]) { im.instanceMatrix.needsUpdate = true; im.computeBoundingSphere(); }
  // wires: thin tubes along their sampled points (contact wires every 2 m, span wires in three points)
  const tubes: THREE.BufferGeometry[] = [];
  for (const w of data.wires ?? []) {
    if (w.p.length < 2) continue;
    const curve = new THREE.CatmullRomCurve3(w.p.map(([x, y, z]) => new THREE.Vector3(x, z, -y)));
    const g = new THREE.TubeGeometry(curve, Math.max(2, w.p.length * (w.p.length < 4 ? 4 : 1)), w.r, 4, false); g.deleteAttribute('uv'); tubes.push(g);
  }
  if (tubes.length) { const wm = new THREE.Mesh(mergeGeometries(tubes), wire); wm.name = 'overhead__wires'; wm.castShadow = false; wm.receiveShadow = false; group.add(wm); tubes.forEach(t => t.dispose()); }
  group.updateMatrixWorld(true); group.traverse(o => { o.matrixAutoUpdate = false; });
  return {group, materials: [steel, concrete, head, lens, wire], circles, poles: poles.length, wires: data.wires?.length ?? 0};
}
