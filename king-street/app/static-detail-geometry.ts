import * as THREE from 'three';
import {sceneCpuSlice} from './scene-flat.ts';

/** Share exact geometry work between ordinary synchronous and live sliced callers. */
export function finishStaticSteps<T>(steps:Generator<void,T>):T {
 let next=steps.next();while(!next.done)next=steps.next();return next.value as T;
}
export async function finishStaticStepsAsync<T>(steps:Generator<void,T>,yieldCpu:()=>Promise<void>):Promise<T> {
 const slice=sceneCpuSlice(yieldCpu);let next=steps.next();
 try{while(!next.done){const pause=slice();if(pause)await pause;next=steps.next();}return next.value as T;}
 finally{steps.return(undefined as T);}
}


// A render batch and its pickable source meshes keep independent attributes,
// counts and indices, but need only one copy of their vertex storage. Track the
// views weakly so a later depth merge can move the same storage once more.
const attributeViews = new WeakMap<THREE.BufferAttribute, {attribute: THREE.BufferAttribute; offset: number}[]>();
export function compactStaticIndex(geometry: THREE.BufferGeometry) {
  const index = geometry.index;
  if (!(index?.array instanceof Uint32Array) || geometry.attributes.position.count > 65535) return;
  // Confirm rather than truncate malformed or nonstandard legacy exports.
  if (index.array.some(value => value > 65535)) return;
  index.array = new Uint16Array(index.array);
  index.needsUpdate = true;
}

export function shareStaticAttribute(source: THREE.BufferAttribute, merged: THREE.BufferAttribute, offset: number) {
  if (source === merged || source.array.constructor !== merged.array.constructor ||
      source.itemSize !== merged.itemSize || source.normalized !== merged.normalized) return;
  const move = (attribute: THREE.BufferAttribute, start: number) => {
    attribute.array = merged.array.subarray(start, start + attribute.count * attribute.itemSize);
    attribute.needsUpdate = true;
    for (const view of attributeViews.get(attribute) ?? []) move(view.attribute, start + view.offset);
  };
  move(source, offset);
  const views = attributeViews.get(merged) ?? [];
  views.push({attribute: source, offset});
  attributeViews.set(merged, views);
}

/** Split an area-wide indexed mesh at triangle boundaries. Each cell has its own
 * index buffer and exact conservative bounds. Vertex buffers remain shared, so
 * a ground mesh spanning a whole city is uploaded only once. */
export function tileLargeMesh(mesh: THREE.Mesh, worldBox: THREE.Box3): THREE.BufferGeometry[] | null {
  const g = mesh.geometry, position = g.attributes.position, src = g.index;
  const width = worldBox.max.x - worldBox.min.x, depth = worldBox.max.z - worldBox.min.z;
  if (!src || src.count < 180000 || Math.max(width, depth) < 150 || src.count % 3) return null;
  const cellSize = Math.max(64, width / 40, depth / 40);
  const columns = Math.floor(width / cellSize) + 1, rows = Math.floor(depth / cellSize) + 1;
  const counts = new Uint32Array(columns * rows), addresses = new Uint16Array(src.count / 3);
  const bounds: (THREE.Box3 | undefined)[] = [];
  const a = new THREE.Vector3(), b = new THREE.Vector3(), c = new THREE.Vector3(), center = new THREE.Vector3();
  for (let i = 0, t = 0; i < src.count; i += 3, t++) {
    a.fromBufferAttribute(position, src.getX(i));
    b.fromBufferAttribute(position, src.getX(i + 1));
    c.fromBufferAttribute(position, src.getX(i + 2));
    center.copy(a).add(b).add(c).multiplyScalar(1 / 3).applyMatrix4(mesh.matrixWorld);
    const x = THREE.MathUtils.clamp(Math.floor((center.x - worldBox.min.x) / cellSize), 0, columns - 1);
    const z = THREE.MathUtils.clamp(Math.floor((center.z - worldBox.min.z) / cellSize), 0, rows - 1);
    const id = z * columns + x;
    counts[id] += 3;
    addresses[t] = id;
    const box = bounds[id] ?? (bounds[id] = new THREE.Box3());
    box.expandByPoint(a).expandByPoint(b).expandByPoint(c);
  }
  if (bounds.filter(Boolean).length < 2) return null;
  const cursor = new Uint32Array(counts.length);
  let end = 0;
  for (let cell = 0; cell < counts.length; cell++) { cursor[cell] = end; end += counts[cell]; }
  const packed = position.count > 65535 ? new Uint32Array(end) : new Uint16Array(end);
  for (let i = 0, t = 0; i < src.count; i += 3, t++) {
    const cell = addresses[t], j = cursor[cell];
    packed[j] = src.getX(i); packed[j + 1] = src.getX(i + 1); packed[j + 2] = src.getX(i + 2);
    cursor[cell] += 3;
  }
  const tiles: THREE.BufferGeometry[] = [];
  for (let cell = 0; cell < counts.length; cell++) {
    const box = bounds[cell];
    if (!box) continue;
    const tile = new THREE.BufferGeometry();
    for (const [name, attribute] of Object.entries(g.attributes)) tile.setAttribute(name, attribute);
    tile.setIndex(new THREE.BufferAttribute(packed.subarray(cursor[cell] - counts[cell], cursor[cell]), 1));
    tile.boundingBox = box;
    tile.boundingSphere = box.getBoundingSphere(new THREE.Sphere());
    tiles.push(tile);
  }
  return tiles;
}

/** Build missing normals on the exporter's tightly packed triangle surfaces.
 * Keep Three's area weighting, float32 accumulation and zero-area handling.
 * Other attribute layouts keep the library's accessor-aware path. */
function* geometryNormalSteps(geometry: THREE.BufferGeometry):Generator<void,void> {
  if (geometry.hasAttribute('normal')) return;
  const positions = geometry.getAttribute('position'), indices = geometry.index;
  if (!(positions instanceof THREE.BufferAttribute) || positions.itemSize !== 3 ||
      !(positions.array instanceof Float32Array) || positions.normalized ||
      indices?.normalized || (indices?.count ?? positions.count) % 3 !== 0) {
    geometry.computeVertexNormals();
    return;
  }

  const p = positions.array, index = indices?.array, count = indices?.count ?? positions.count;
  const normals = new Float32Array(positions.count * 3);
  for (let i = 0; i < count; i += 3) {
    if(i%12288===0)yield;
    const a = (index ? index[i] : i) * 3;
    const b = (index ? index[i + 1] : i + 1) * 3;
    const c = (index ? index[i + 2] : i + 2) * 3;
    const cx = p[c] - p[b], cy = p[c + 1] - p[b + 1], cz = p[c + 2] - p[b + 2];
    const ax = p[a] - p[b], ay = p[a + 1] - p[b + 1], az = p[a + 2] - p[b + 2];
    const x = cy * az - cz * ay, y = cz * ax - cx * az, z = cx * ay - cy * ax;
    normals[a] += x; normals[a + 1] += y; normals[a + 2] += z;
    normals[b] += x; normals[b + 1] += y; normals[b + 2] += z;
    normals[c] += x; normals[c + 1] += y; normals[c + 2] += z;
  }
  for (let i = 0; i < normals.length; i += 3) {
    if(i%12288===0)yield;
    const x = normals[i], y = normals[i + 1], z = normals[i + 2];
    const scale = 1 / (Math.sqrt(x * x + y * y + z * z) || 1);
    normals[i] *= scale; normals[i + 1] *= scale; normals[i + 2] *= scale;
  }
  geometry.setAttribute('normal', new THREE.BufferAttribute(normals, 3));
}

/** Static meshes require both bounds for shadows, picking and culling. Scan
 * dense positions twice for the exact box and sphere. The stock calls scan
 * three times because computeBoundingSphere builds another temporary box. */
export function* staticBoundsSteps(geometry: THREE.BufferGeometry):Generator<void,void> {
  const position = geometry.getAttribute('position');
  if (!(position instanceof THREE.BufferAttribute) || position.itemSize !== 3 ||
      position.normalized || !(position.array instanceof Float32Array) || geometry.morphAttributes.position) {
    geometry.computeBoundingBox();
    geometry.computeBoundingSphere();
    return;
  }
  const p = position.array;
  let xmin = Infinity, ymin = Infinity, zmin = Infinity, xmax = -Infinity, ymax = -Infinity, zmax = -Infinity;
  for (let i = 0; i < p.length; i += 3) {
    if(i%12288===0)yield;
    const x = p[i], y = p[i + 1], z = p[i + 2];
    xmin = Math.min(xmin, x); ymin = Math.min(ymin, y); zmin = Math.min(zmin, z);
    xmax = Math.max(xmax, x); ymax = Math.max(ymax, y); zmax = Math.max(zmax, z);
  }
  const box = new THREE.Box3(new THREE.Vector3(xmin, ymin, zmin), new THREE.Vector3(xmax, ymax, zmax));
  const sphere = new THREE.Sphere();
  box.getCenter(sphere.center);
  let largest = 0;
  for (let i = 0; i < p.length; i += 3) {
    if(i%12288===0)yield;
    const x = sphere.center.x - p[i], y = sphere.center.y - p[i + 1], z = sphere.center.z - p[i + 2];
    largest = Math.max(largest, x * x + y * y + z * z);
  }
  sphere.radius = Math.sqrt(largest);
  geometry.boundingSphere = sphere;
  geometry.boundingBox = box;
}

export function ensureGeometryNormals(geometry:THREE.BufferGeometry) {
 finishStaticSteps(geometryNormalSteps(geometry));
}
export function ensureGeometryNormalsAsync(geometry:THREE.BufferGeometry,yieldCpu:()=>Promise<void>) {
 return finishStaticStepsAsync(geometryNormalSteps(geometry),yieldCpu);
}

/** Keep the same surface and lighting directions in compact GPU attributes.
 * Axis-aligned normals fit bytes exactly. Other unit vectors lose at most
 * half a signed 16-bit step per component. Leave nonstandard inputs alone. */
export function* packStaticNormals(geometries:Iterable<THREE.BufferGeometry>):Generator<void,void> {
 const packed=new WeakMap<THREE.BufferAttribute,THREE.BufferAttribute>();
 for(const geometry of geometries){
  const attribute=geometry.getAttribute('normal');
  if(!(attribute instanceof THREE.BufferAttribute)||attribute.itemSize!==3||attribute.normalized||
     !(attribute.array instanceof Float32Array)||geometry.morphAttributes.normal)continue;
  const existing=packed.get(attribute);
  if(existing){geometry.setAttribute('normal',existing);continue;}
  const source=attribute.array;
  // Use one layout for small meshes so material batches stay together.
  let axis=attribute.count>=65536,valid=true;
  for(let i=0;i<source.length;i++){
   if(i%16384===0)yield;
   const value=source[i];
   if(!(value>=-1&&value<=1)){valid=false;break;}
   if(value!==0&&value!==1&&value!==-1)axis=false;
  }
  if(!valid)continue;
  const array=axis?new Int8Array(source.length):new Int16Array(source.length);
  const scale=axis?127:32767;
  for(let i=0;i<source.length;i++){
   if(i%16384===0)yield;
   array[i]=Math.round(source[i]*scale);
  }
  const compact=new THREE.BufferAttribute(array,3,true);
  packed.set(attribute,compact);geometry.setAttribute('normal',compact);
 }
}
export function computeStaticBounds(geometry:THREE.BufferGeometry) {
 finishStaticSteps(staticBoundsSteps(geometry));
}
