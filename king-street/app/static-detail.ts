import * as THREE from 'three';
import {compactStaticIndex, staticBoundsSteps, finishStaticSteps, finishStaticStepsAsync, shareStaticAttribute, tileLargeMesh} from './static-detail-geometry.ts';
import {mergeAttributes, mergeGeometries} from 'three/addons/utils/BufferGeometryUtils.js';

type LayerOf = (mesh: THREE.Object3D) => string;

export function evidenceTint(layer: string, evidence: unknown) {
  if (layer === 'roofs' || layer === 'terrain') return '#56b3c4';
  return evidence === 'visual-fit' ? '#7caf8c' : '#d2a25e';
}

/** Generated paint swatches differ only by a uniform tint. Preserve that linear
 * colour in a vertex attribute so walls of different colours can share a draw.
 * Match the full material state after depth priorities and finishes are set. */
export function batchPaintSwatches(meshes: THREE.Mesh[], layerOf: LayerOf, originals: Map<THREE.Mesh, THREE.Material | THREE.Material[]>) {
  const users = new Map<THREE.MeshStandardMaterial, THREE.Mesh[]>();
  const blocked = new Set<THREE.Material>();
  // A source geometry shared with another material must keep its original attributes.
  const owner = new Map<THREE.BufferGeometry, THREE.Material | THREE.Material[]>();
  for (const mesh of meshes) {
    const material = mesh.material, previous = owner.get(mesh.geometry);
    if (previous && previous !== material) {
      for (const m of ([] as THREE.Material[]).concat(previous, material)) blocked.add(m);
    } else owner.set(mesh.geometry, material);
    if (Array.isArray(material)) { for (const m of material) blocked.add(m); continue; }
    if (!(material instanceof THREE.MeshStandardMaterial) || !/^#[\da-f]{6}$/i.test(material.name)) continue;
    if (!['facades', 'buildings', 'roofs'].includes(layerOf(mesh)) || mesh.geometry.hasAttribute('color')) {
      blocked.add(material);
      continue;
    }
    const list = users.get(material);
    if (list) list.push(mesh); else users.set(material, [mesh]);
  }
  const groups = new Map<string, THREE.MeshStandardMaterial[]>();
  for (const material of users.keys()) {
    if (blocked.has(material) || material.constructor !== THREE.MeshStandardMaterial || material.vertexColors ||
        material.transparent || material.clippingPlanes?.length || Object.values(material).some(v => v instanceof THREE.Texture)) continue;
    const {uuid, name, color, metadata, ...state} = material.toJSON();
    const key = JSON.stringify([state, (material as THREE.MeshStandardMaterial & {defines?: object}).defines, material.customProgramCacheKey()]);
    const list = groups.get(key);
    if (list) list.push(material); else groups.set(key, [material]);
  }
  const retired: THREE.Material[] = [];
  for (const group of groups.values()) {
    if (group.length < 2) continue;
    const shared = group[0];
    for (const material of group) {
      const rgb = material.color.toArray(), bytes = rgb.map(v => Math.round(v * 255));
      const exactBytes = bytes.every((b, i) => b >= 0 && b <= 255 && Math.fround(b / 255) === Math.fround(rgb[i]));
      for (const mesh of users.get(material)!) {
        const g = mesh.geometry;
        if (!g.hasAttribute('color')) {
          const count = g.attributes.position.count;
          const colors = exactBytes ? new Uint8Array(count * 3) : new Float32Array(count * 3);
          const values = exactBytes ? bytes : rgb;
          // Tile one colour with typed-array copies, including very large shells.
          colors.set(values);
          for (let size = 3; size < colors.length; size *= 2) colors.copyWithin(size, 0, Math.min(size, colors.length - size));
          g.setAttribute('color', new THREE.BufferAttribute(colors, 3, exactBytes));
        }
        mesh.material = shared;
        originals.set(mesh, shared);
      }
      if (material !== shared) retired.push(material);
    }
    shared.color.setRGB(1, 1, 1);
    shared.vertexColors = true;
    shared.needsUpdate = true;
  }
  for (const material of retired) material.dispose();
}

type Part = {mesh: THREE.Mesh; x: number; z: number; triangles: number};

/** Keep a sparse material together, subdividing costly batches until their
 * contents can be culled at roughly street-block scale. All cameras and shadow
 * frustums use Three's ordinary bounds tests. No triangles change. */
function partition(parts: Part[], size: number, out: THREE.Mesh[][]) {
  let triangles = 0;
  for (const part of parts) triangles += part.triangles;
  if (parts.length < 2 || triangles <= 14000 || size <= 38) {
    out.push(parts.map(p => p.mesh));
    return;
  }
  const half = size / 2, children = new Map<string, Part[]>();
  for (const part of parts) {
    const key = Math.floor(part.x / half) + ',' + Math.floor(part.z / half);
    const group = children.get(key);
    if (group) group.push(part); else children.set(key, [part]);
  }
  for (const group of children.values()) partition(group, half, out);
}

function* mergeStaticGeometry(meshes: THREE.Mesh[]):Generator<void,{geometry:THREE.BufferGeometry;matrix:THREE.Matrix4|null}|null> {
  const first = meshes[0];
  // The generator stores buildings in one coordinate frame. Keeping that frame
  // avoids materialising and rotating a temporary copy of every vertex.
  const common = meshes.every(o => o.matrixWorld.equals(first.matrixWorld));
  const parts = meshes.map(o => common ? o.geometry : o.geometry.clone().applyMatrix4(o.matrixWorld));
  let geometry: THREE.BufferGeometry | null;
  if (parts.some(g => Object.keys(g.morphAttributes).length)) {
    geometry = mergeGeometries(parts, false);
  } else {
    geometry = new THREE.BufferGeometry();
    for (const name of Object.keys(parts[0].attributes)) {
      yield;
      const attribute = mergeAttributes(parts.map(g => g.attributes[name] as THREE.BufferAttribute));
      if (!attribute) { geometry.dispose(); geometry = null; break; }
      geometry.setAttribute(name, attribute);
    }
    if (geometry && first.geometry.index) {
      const count = parts.reduce((n, g) => n + g.index!.count, 0);
      const indices = geometry.attributes.position.count > 65535 ? new Uint32Array(count) : new Uint16Array(count);
      let vertex = 0, offset = 0;
      for (const g of parts) {
        const src = g.index!.array;
        for (let i = 0; i < src.length; i++) {if(i%16384===0)yield;indices[offset++] = vertex + src[i];}
        vertex += g.attributes.position.count;
      }
      geometry.setIndex(new THREE.BufferAttribute(indices, 1));
    }
  }
  if (!common) for (const g of parts) g.dispose();
  if (geometry) {
    yield* staticBoundsSteps(geometry);
    // Keep the original meshes usable by selection, collision and evidence,
    // without retaining a second city's worth of position/normal/UV buffers.
    // A transformed merge keeps its sources unchanged in their local frames.
    if (common) {
      for (const [name, merged] of Object.entries(geometry.attributes)) {
        let offset = 0;
        for (const part of parts) {
          const source = part.attributes[name];
          if (source instanceof THREE.BufferAttribute && merged instanceof THREE.BufferAttribute) shareStaticAttribute(source, merged, offset);
          offset += source.count * source.itemSize;
        }
      }
    }
  }
  return geometry ? {geometry, matrix: common ? first.matrixWorld : null} : null;
}

function* batchStaticWorldSteps(objects: THREE.Mesh[], layerOf: LayerOf, shadowClass: (mesh: THREE.Mesh) => string, root: THREE.Group):Generator<void,THREE.Mesh[]> {
  const cells = new Map<string, Part[]>(), box = new THREE.Box3(), center = new THREE.Vector3();
  const batches: THREE.Mesh[] = [];
  function addBatch(list: THREE.Mesh[], geometry: THREE.BufferGeometry, matrix: THREE.Matrix4 | null, tiled = false) {
    const first = list[0], batch = new THREE.Mesh(geometry, first.material);
    if (matrix) {
      batch.matrixAutoUpdate = false;
      batch.matrix.copy(matrix);
      batch.matrix.decompose(batch.position, batch.quaternion, batch.scale);
    }
    batch.name = 'batch|' + layerOf(first) + '|' + batch.id;
    batch.castShadow = first.castShadow;
    batch.receiveShadow = first.receiveShadow;
    batch.userData = {layer: layerOf(first), evidence: first.userData.evidence, original: first.material, members: list, sharedTile: tiled};
    root.add(batch);
    batches.push(batch);
    for (const o of list) { o.visible = false; o.userData.batched = batch; }
  }
  for (const mesh of objects) {
    yield;
    compactStaticIndex(mesh.geometry);
    const layer = layerOf(mesh);
    if (Array.isArray(mesh.material) || !mesh.visible || (layer === 'vegetation' && /___leaf/.test(mesh.name))) continue;
    const g = mesh.geometry;
    if (g.drawRange.start !== 0 || Number.isFinite(g.drawRange.count) || Object.keys(g.morphAttributes).length ||
        (mesh as THREE.SkinnedMesh).isSkinnedMesh || (mesh as THREE.InstancedMesh).isInstancedMesh) continue;
    mesh.updateWorldMatrix(true, false);
    if (!g.boundingBox) g.computeBoundingBox();
    box.copy(g.boundingBox!).applyMatrix4(mesh.matrixWorld);
    const tiles = tileLargeMesh(mesh, box);
    if (tiles) {
      for (const geometry of tiles) addBatch([mesh], geometry, mesh.matrixWorld, true);
      continue;
    }
    box.getCenter(center);
    const layout = Object.entries(g.attributes).sort(([a], [b]) => a.localeCompare(b)).map(([name, a]) =>
      [name, a.array.constructor.name, a.itemSize, a.normalized].join(':')).join(',');
    const key = [layer, mesh.material.uuid, g.index ? 1 : 0, layout, mesh.castShadow, mesh.receiveShadow, shadowClass(mesh),
      Math.floor(center.x / 300), Math.floor(center.z / 300)].join('|');
    const part = {mesh, x: center.x, z: center.z, triangles: (g.index?.count ?? g.attributes.position.count) / 3};
    const group = cells.get(key);
    if (group) group.push(part); else cells.set(key, [part]);
  }
  for (const parts of cells.values()) {
    const lists: THREE.Mesh[][] = [];
    partition(parts, 300, lists);
    for (const list of lists) {
      if (list.length < 2) continue;
      yield;
      const merged = yield* mergeStaticGeometry(list);
      if (!merged) continue;
      addBatch(list, merged.geometry, merged.matrix);
    }
  }
  return batches;
}

export function batchStaticWorld(objects:THREE.Mesh[],layerOf:LayerOf,shadowClass:(mesh:THREE.Mesh)=>string,root:THREE.Group) {
 return finishStaticSteps(batchStaticWorldSteps(objects,layerOf,shadowClass,root));
}
export function batchStaticWorldAsync(objects:THREE.Mesh[],layerOf:LayerOf,shadowClass:(mesh:THREE.Mesh)=>string,root:THREE.Group,yieldCpu:()=>Promise<void>) {
 return finishStaticStepsAsync(batchStaticWorldSteps(objects,layerOf,shadowClass,root),yieldCpu);
}
