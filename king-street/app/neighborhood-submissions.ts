import * as THREE from 'three';
import {shareStaticAttribute} from './static-detail-geometry.ts';

type Mesh = THREE.Mesh<THREE.BufferGeometry, THREE.Material | THREE.Material[]>;
type Group = {mesh: THREE.BatchedMesh; sources: Mesh[]; ids: number[]; visible: number};

/** Multi-draw submission only: original meshes keep geometry, owner, visibility,
 * picking and evidence state. No source geometry or cell ownership is merged.
 * Evidence mode falls back to those original submissions unchanged. */
export function batchNeighborhoodSubmissions(
  sources: Mesh[], root: THREE.Group,
  options: {supported: boolean; layerOf: (o: THREE.Object3D) => string; maxBytes?: number},
) {
  const groups: Group[] = [], controllers = new Set<Mesh>();
  const fallback = new THREE.Group(); fallback.name = 'neighborhood-submission-fallback';
  const physical = new THREE.Group(); physical.name = 'neighborhood-multi-draw';
  root.add(physical, fallback);
  let enabled = options.supported, disposed = false;
  const stats = {supported: options.supported, enabled, sourceMeshes: 0, submissions: 0,
    geometryBytes: 0, sourceGeometryBytes: 0, retiredBakeGeometries: 0, retiredBakeBytes: 0, instancesVisible: 0, syncMs: 0};
  const original = (o: Mesh): Mesh => (o.userData.members as Mesh[] | undefined)?.[0] ?? o;
  const keyOf = (o: Mesh) => {
    const source = original(o), geometry = o.geometry, material = o.material;
    if (Array.isArray(material) || material.transparent || !material.visible ||
        source.userData.class !== 'distant_exterior' || !source.userData.bakedAppearance ||
        !['buildings', 'facades', 'roofs'].includes(options.layerOf(o)) ||
        !o.visible || (o as THREE.BatchedMesh).isBatchedMesh ||
        (o as THREE.InstancedMesh).isInstancedMesh || (o as THREE.SkinnedMesh).isSkinnedMesh ||
        geometry.drawRange.start !== 0 || Number.isFinite(geometry.drawRange.count) ||
        geometry.groups.length || Object.keys(geometry.morphAttributes).length ||
        !geometry.attributes.position) return null;
    const layout = Object.entries(geometry.attributes).sort(([a], [b]) => a.localeCompare(b)).map(([name, a]) => {
      if ((a as THREE.InterleavedBufferAttribute).isInterleavedBufferAttribute || a.count !== geometry.attributes.position.count) return null;
      return [name, a.array.constructor.name, a.itemSize, a.normalized].join(':');
    });
    if (layout.includes(null)) return null;
    o.updateWorldMatrix(true, false);
    // Native BatchedMesh does not support mirrored instance transforms.
    if (!(o.matrixWorld.determinant() > 0)) return null;
    return [material.uuid, options.layerOf(o), !!geometry.index, layout.join(';'), o.castShadow, o.receiveShadow,
      source.userData.evidence ?? '', source.userData.class].join('|');
  };
  const byteSize = (g: THREE.BufferGeometry) => Object.values(g.attributes).reduce((n, a) => n+a.array.byteLength, 0)+(g.index?.array.byteLength ?? 0);
  const capacityBytes = (g: THREE.BufferGeometry) => Object.values(g.attributes).reduce((n, a) => n+a.array.byteLength, 0)+(g.index?.count ?? 0)*4;
  const plans = new Map<string, {sources: Mesh[]; bytes: number}[]>();
  if (options.supported) for (const source of sources) {
    const key = keyOf(source); if (key === null) continue;
    const size = capacityBytes(source.geometry), maxBytes = options.maxBytes ?? 32*1024*1024;
    if (size > maxBytes) continue;
    let lists = plans.get(key); if (!lists) {lists = []; plans.set(key, lists);}
    let list = lists[lists.length-1];
    if (!list || list.bytes+size > maxBytes) {list = {sources: [], bytes: 0}; lists.push(list);}
    list.sources.push(source); list.bytes += size;
  }
  for (const lists of plans.values()) for (const {sources: list} of lists) {
    if (list.length < 2) continue;
    const vertices = list.reduce((n, m) => n+m.geometry.attributes.position.count, 0);
    const indices = list.reduce((n, m) => n+(m.geometry.index?.count ?? 0), 0);
    const first = list[0];
    const batch = new THREE.BatchedMesh(list.length, vertices, indices, first.material as THREE.Material);
    batch.name = 'neighborhood_multi_draw__'+options.layerOf(first);
    batch.castShadow = first.castShadow; batch.receiveShadow = first.receiveShadow;
    batch.sortObjects = false; batch.perObjectFrustumCulled = true;
    // Native per-instance bounds are authoritative for every camera, including shadows.
    batch.frustumCulled = false;
    batch.userData = {layer: options.layerOf(first), class: 'distant_exterior', bakedAppearance: true,
      submissionBatch: true, members: list.flatMap(m => (m.userData.members as Mesh[] | undefined) ?? [m])};
    const group: Group = {mesh: batch, sources: list, ids: [], visible: list.length};
    for (const source of list) {
      source.updateWorldMatrix(true, false);
      const world = source.matrixWorld.clone();
      const geometry = batch.addGeometry(source.geometry), instance = batch.addInstance(geometry);
      batch.setMatrixAt(instance, world); group.ids.push(instance);
      const range = batch.getGeometryRangeAt(geometry);
      if (range) for (const [name, attribute] of Object.entries(source.geometry.attributes)) {
        shareStaticAttribute(attribute as THREE.BufferAttribute, batch.geometry.attributes[name] as THREE.BufferAttribute,
          range.vertexStart * attribute.itemSize);
      }
      // Preserve world-space picking/baking when a controller is detached.
      source.removeFromParent(); source.matrixAutoUpdate = false;
      source.matrix.copy(world); source.matrixWorld.copy(world);
      source.matrix.decompose(source.position, source.quaternion, source.scale);
      source.userData.submissionController = true; controllers.add(source);
      stats.sourceGeometryBytes += byteSize(source.geometry);
    }
    const nativeBefore = batch.onBeforeRender;
    batch.onBeforeRender = function (renderer, scene, camera, geometry, material, ...rest) {
      syncGroup(group, renderer, scene, camera);
      nativeBefore.call(this, renderer, scene, camera, geometry, material, ...rest);
    };
    physical.add(batch); groups.push(group);
    stats.instancesVisible += group.visible;
    stats.geometryBytes += byteSize(batch.geometry);
  }
  stats.sourceMeshes = controllers.size; stats.submissions = groups.length;

  function syncGroup(group: Group, renderer?: THREE.WebGLRenderer, scene?: THREE.Scene, camera?: THREE.Camera) {
    const started = performance.now(); let visible = 0;
    for (let i = 0; i < group.sources.length; i++) {
      const source = group.sources[i]; let show = source.visible;
      // Preserve camera-specific conservative depth decisions without changing
      // source visibility for reflection or shadow cameras. Existing callbacks
      // temporarily zero drawRange and restore it in onAfterRender.
      if (show && renderer && scene && camera) {
        const material = Array.isArray(source.material) ? source.material[0] : source.material;
        // Three passes null for ungrouped meshes despite the callback's Group type.
        source.onBeforeRender(renderer, scene, camera, source.geometry, material, null as unknown as THREE.Group);
        show = source.geometry.drawRange.count !== 0;
        source.onAfterRender(renderer, scene, camera, source.geometry, material, null as unknown as THREE.Group);
      }
      if (group.mesh.getVisibleAt(group.ids[i]) !== show) group.mesh.setVisibleAt(group.ids[i], show);
      if (show) visible++;
    }
    stats.instancesVisible += visible-group.visible; group.visible = visible;
    stats.syncMs = performance.now()-started;
  }
  function setEnabled(on: boolean) {
    if (disposed) return;
    enabled = on && options.supported; stats.enabled = enabled;
    physical.visible = enabled;
    for (const source of controllers) {
      if (enabled) source.removeFromParent(); else fallback.add(source);
    }
  }
  setEnabled(enabled);
  return {
    meshes: groups.map(g => g.mesh), controllers, stats,
    get enabled() {return enabled;},
    setEnabled,
    releaseBakeBuffers(scene: THREE.Scene) {
      if (!enabled || !controllers.size) return;
      // The lighting bakes draw the individual exteriors before regular
      // multi-draw rendering begins. Dispose their WebGL copies and VAOs;
      // Three can upload the unchanged CPU geometry again for evidence mode.
      // Disposing a shared BufferAttribute would invalidate another mesh's VAO.
      const kept = new Set<THREE.BufferGeometry>(), buffers = new Set<object>();
      const backing = (a: THREE.BufferAttribute | THREE.InterleavedBufferAttribute) =>
        'data' in a ? a.data : a;
      scene.traverse(object => {
        const g = (object as Mesh).geometry;
        if (!g || kept.has(g)) return;
        kept.add(g);
        for (const a of Object.values(g.attributes)) buffers.add(backing(a));
        if (g.index) buffers.add(g.index);
      });
      const retired = new Set<THREE.BufferGeometry>();
      for (const mesh of controllers) {
        const g = mesh.geometry;
        if (retired.has(g) || kept.has(g) || (g.index && buffers.has(g.index)) ||
            Object.values(g.attributes).some(a => buffers.has(backing(a)))) continue;
        retired.add(g);
        g.dispose();
        stats.retiredBakeGeometries++;
        stats.retiredBakeBytes += byteSize(g);
      }
    },
    sync() {for (const group of groups) syncGroup(group);},
    dispose() {
      if (disposed) return; disposed = true;
      for (const group of groups) group.mesh.dispose();
      for (const source of controllers) {source.removeFromParent(); delete source.userData.submissionController;}
      physical.removeFromParent(); fallback.removeFromParent();
    },
  };
}

export type NeighborhoodSubmissions = ReturnType<typeof batchNeighborhoodSubmissions>;
