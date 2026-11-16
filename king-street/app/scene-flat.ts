import {Box3, BufferAttribute, BufferGeometry, Matrix4, Mesh, Sphere, Vector3} from 'three';
import type {GLTFParser} from 'three/addons/loaders/GLTFLoader.js';

type Accessor = {
  type: string; componentType: number; bufferView: number; byteOffset?: number;
  count: number; normalized?: boolean; sparse?: unknown; min?: number[]; max?: number[];
};
type Primitive = {mode?: number; material: number; indices: number; attributes: Record<string, number>};
type StaticDocument = {
  nodes: {name: string; mesh: number; matrix: number[]; extras?: object}[];
  meshes: {primitives: Primitive[]; extras?: object}[];
  scenes: {nodes: number[]}[]; materials: unknown[]; accessors: Accessor[];
  bufferViews: {byteStride?: number}[]; animations?: unknown[]; skins?: unknown[]; cameras?: unknown[];
};

const arrays: Record<number, Float32ArrayConstructor | Uint32ArrayConstructor | Uint16ArrayConstructor | Uint8ArrayConstructor> =
  {5126: Float32Array, 5125: Uint32Array, 5123: Uint16Array, 5121: Uint8Array};
const names: Record<string, string> = {POSITION: 'position', NORMAL: 'normal', TEXCOORD_0: 'uv', COLOR_0: 'color'};
const sizes: Record<string, number> = {SCALAR: 1, VEC2: 2, VEC3: 3, VEC4: 4};

// Accept only the static exporter's flat, one-mesh-per-node layout. All
// hierarchical models, instancing, animations and general glTF features keep
// GLTFLoader's regular path. Its material and texture loaders serve both paths.
export function flatLayout(doc: StaticDocument) {
  if (!doc.nodes?.length || !doc.meshes || !doc.materials || !doc.accessors || !doc.bufferViews ||
      doc.scenes?.length !== 1 || !doc.scenes[0].nodes ||
      doc.animations?.length || doc.cameras?.length || doc.skins?.length) return false;
  const roots = new Set(doc.scenes[0].nodes);
  if (roots.size !== doc.scenes[0].nodes.length || [...roots].some(i => !Number.isInteger(i) || i < 0 || i >= doc.nodes.length)) return false;
  const meshes = new Set<number>();
  for (const node of doc.nodes) {
    if (!node.name || node.matrix?.length !== 16 || meshes.has(node.mesh) ||
        Object.keys(node).some(k => !['mesh','name','extras','matrix'].includes(k))) return false;
    meshes.add(node.mesh);
    const mesh = doc.meshes[node.mesh];
    if (mesh?.primitives?.length !== 1 ||
        Object.keys(mesh).some(k => !['primitives','name','extras'].includes(k))) return false;
    const p = mesh.primitives[0];
    if ((p.mode ?? 4) !== 4 || !doc.materials[p.material] ||
        Object.keys(p).some(k => !['mode','indices','attributes','material'].includes(k)) ||
        !p.attributes || Object.keys(p.attributes).some(s => !Object.hasOwn(names, s))) return false;
    const pos = doc.accessors[p.attributes.POSITION], index = doc.accessors[p.indices];
    if (pos?.type !== 'VEC3' || pos.min?.length !== 3 || pos.max?.length !== 3 ||
        !index || index.type !== 'SCALAR' || ![5121,5123,5125].includes(index.componentType)) return false;
  }
  return doc.accessors.every(a => !!arrays[a.componentType] && !!sizes[a.type] &&
    !a.normalized && !a.sparse && !!doc.bufferViews[a.bufferView] && !doc.bufferViews[a.bufferView].byteStride);
}

/** A live caller supplies the scheduling policy. Startup never waits here. */
export function sceneCpuSlice(yieldCpu?: () => Promise<void>, budgetMs = 4) {
  let start = performance.now();
  return () => {
    if (!yieldCpu || performance.now() - start < budgetMs) return null;
    return yieldCpu().then(() => { start = performance.now(); });
  };
}

/** Remove only known static tree roots; retain resource indices and every shared view.
 * Unknown exporter layouts keep the ordinary loader path unchanged. */
export function omitStaticTrees(json: unknown, classes: ReadonlySet<string>): Set<number> {
  const doc = json as StaticDocument & {images?: {bufferView?: number}[]; textures?: {source?: number}[]};
  if (!flatLayout(doc) || Object.keys(doc).some(k => !['asset','scene','scenes','nodes','meshes','materials','accessors','bufferViews','buffers','images','textures','samplers','extras'].includes(k))) return new Set();
  // Undeclared/custom material extensions may hold additional image/accessor references.
  const hasExtensions = (value: unknown): boolean => !!value && typeof value === 'object' && Object.entries(value).some(([k,v]) => k === 'extensions' || hasExtensions(v));
  if (hasExtensions(doc)) return new Set();
  const omitted = new Set(doc.scenes[0].nodes.filter(i => {
    const node = doc.nodes[i], layer = (node.extras as {layer?: string}|undefined)?.layer;
    return node.name.startsWith('vegetation___') && (layer === undefined || layer === 'vegetation') && classes.has(node.name.slice('vegetation___'.length));
  }));
  if (!omitted.size) return new Set();
  const keptAccessors = new Set<number>(), removedAccessors = new Set<number>(), keptMaterials = new Set<number>(), removedMaterials = new Set<number>();
  for (const i of doc.scenes[0].nodes) {
    const p = doc.meshes[doc.nodes[i].mesh].primitives[0], removed = omitted.has(i);
    const accessors = removed ? removedAccessors : keptAccessors;
    accessors.add(p.indices);for (const a of Object.values(p.attributes)) accessors.add(a);
    (removed ? removedMaterials : keptMaterials).add(p.material);
  }
  const keptViews = new Set([...keptAccessors].map(i => doc.accessors[i].bufferView));
  const removedViews = new Set([...removedAccessors].map(i => doc.accessors[i].bufferView));
  const imageViews = (materials: Set<number>, views: Set<number>) => {
    const visit = (value: unknown, key = '') => {
      if (!value || typeof value !== 'object') return;
      if (key.endsWith('Texture')) {
        const index = (value as {index?: number}).index;
        const source = index === undefined ? undefined : doc.textures?.[index]?.source;
        const view = source === undefined ? undefined : doc.images?.[source]?.bufferView;
        if (view !== undefined) views.add(view);
      }
      for (const [k,v] of Object.entries(value)) visit(v,k);
    };
    for (const i of materials) visit(doc.materials[i]);
  };
  imageViews(keptMaterials, keptViews);imageViews(removedMaterials, removedViews);
  // Only views proven exclusively owned by the omitted nodes may be discarded.
  doc.scenes[0].nodes = doc.scenes[0].nodes.filter(i => !omitted.has(i));
  return new Set([...removedViews].filter(i => !keptViews.has(i)));
}

export function flatNodes(parser: GLTFParser, buffers: ArrayBuffer[], binaryReady: Promise<void>, yieldCpu?: () => Promise<void>) {
  const doc = parser.json as StaticDocument;
  if (!flatLayout(doc)) return {};
  const attributes = new Map<number, BufferAttribute>();
  const geometries = new Map<string, BufferGeometry>();
  const matrix = new Matrix4();
  const slice = sceneCpuSlice(yieldCpu);
  let assembly = Promise.resolve();

  function attribute(i: number) {
    let result = attributes.get(i);
    if (!result) {
      const a = doc.accessors[i], width = sizes[a.type];
      const data = new arrays[a.componentType](buffers[a.bufferView], a.byteOffset ?? 0, a.count * width);
      result = new BufferAttribute(data, width);
      attributes.set(i, result);
    }
    return result;
  }
  function geometry(p: Primitive) {
    const a = p.attributes;
    const key = [p.indices, a.POSITION, a.NORMAL, a.TEXCOORD_0, a.COLOR_0, p.mode].join(':');
    let result = geometries.get(key);
    if (!result) {
      result = new BufferGeometry();
      for (const [semantic, i] of Object.entries(a)) result.setAttribute(names[semantic], attribute(i));
      result.setIndex(attribute(p.indices));
      const pos = doc.accessors[a.POSITION];
      result.boundingBox = new Box3(new Vector3().fromArray(pos.min!), new Vector3().fromArray(pos.max!));
      result.boundingSphere = result.boundingBox.getBoundingSphere(new Sphere());
      geometries.set(key, result);
    }
    return result;
  }

  return {afterRoot() {
    attributes.clear();
    geometries.clear();
    return null;
  }, loadNode(index: number) {
    const node = doc.nodes[index], source = doc.meshes[node.mesh], p = source.primitives[0];
    // Reserve names in the scene's node order, before material promises settle.
    const name = parser.createUniqueName(node.name);
    return Promise.all([parser.getDependency('material', p.material), binaryReady]).then(([material]) => {
      const build = () => {
        const mesh = new Mesh(geometry(p), material);
        parser.assignFinalMaterial(mesh);
        mesh.name = name;
        Object.assign(mesh.userData, source.extras, {name: node.name}, node.extras);
        mesh.applyMatrix4(matrix.fromArray(node.matrix));
        const association = {nodes: index, meshes: node.mesh, primitives: 0};
        parser.associations.set(mesh, association);
        return mesh;
      };
      // Preserve the original immediate assembly path for startup.
      if (!yieldCpu) return build();
      const next = assembly.then(async () => {
        const pause = slice();if (pause) await pause;
        return build();
      });
      assembly = next.then(() => {}, () => {});
      return next;
    });
  }};
}
