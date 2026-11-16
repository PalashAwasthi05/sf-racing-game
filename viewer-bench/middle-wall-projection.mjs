// CPU-only production controller selection and target-wall geometry projection; no rendering/GPU.
import assert from 'node:assert/strict';
import {readFile,writeFile} from 'node:fs/promises';
import {gunzipSync} from 'node:zlib';
import path from 'node:path';
import * as THREE from '../king-street/node_modules/three/build/three.module.js';
import {createFacadeMiddleController,validateFacadeMiddleIndex,decodeFacadeBatch} from '../king-street/app/facade-middle.ts';
import {coverageRange,drawTier} from '../king-street/app/stream-transition.ts';
import {frontageScreenFraction} from '../king-street/app/frontage-projection.ts';
const root=path.resolve(import.meta.dirname,'../..'),publicDir=path.join(root,'work/king-street/public');
const out=path.join(root,'agent-handoff/reviews/2026-11-12/clarity');
const json=async p=>JSON.parse(await readFile(p,'utf8'));
const manifest=await json(path.join(publicDir,'compare/neighborhood-stream.json')),world=await json(path.join(publicDir,manifest.world)),baseline=await json(path.join(out,'baseline/report.json'));
const middleIndex=validateFacadeMiddleIndex(await json(path.join(publicDir,manifest.facadeMiddle)),manifest);
const pose=baseline.camera,camera=new THREE.PerspectiveCamera(pose.fov,1920/1080,pose.near,pose.far);camera.position.fromArray(pose.position);camera.quaternion.fromArray(pose.quaternion);camera.updateMatrixWorld();
const clip=new THREE.Matrix4().multiplyMatrices(camera.projectionMatrix,camera.matrixWorldInverse),frustum=new THREE.Frustum().setFromProjectionMatrix(clip),direction=new THREE.Vector3();camera.getWorldDirection(direction);
// Base header fits in its first independently compressed 8MiB chunk. Bounds are real accessor bounds.
const first=gunzipSync(await readFile(path.join(publicDir,manifest.base.chunks[0]))),length=first.readUInt32LE(12);assert.ok(length+20<=first.length);
const base=JSON.parse(first.subarray(20,20+length).toString()),buildings=new Map(world.buildings.map(b=>[b.id,b])),footprints=new Map();
function visit(index,parent=new THREE.Matrix4()){
 const node=base.nodes[index],local=node.matrix?new THREE.Matrix4().fromArray(node.matrix):new THREE.Matrix4().compose(new THREE.Vector3().fromArray(node.translation??[0,0,0]),new THREE.Quaternion().fromArray(node.rotation??[0,0,0,1]),new THREE.Vector3().fromArray(node.scale??[1,1,1]));
 const matrix=new THREE.Matrix4().multiplyMatrices(parent,local),layer=node.extras?.layer??node.name?.split('___')[0].split('__')[0],id=node.extras?.featureId??node.name?.split('___')[0].split('__')[1];
 if((layer==='foundations'||node.extras?.class==='distant_exterior')&&buildings.has(id)&&node.mesh!==undefined){
  for(const primitive of base.meshes[node.mesh].primitives){const accessor=base.accessors[primitive.attributes.POSITION];if(!accessor.min||!accessor.max)continue;
   const box=new THREE.Box3(new THREE.Vector3().fromArray(accessor.min),new THREE.Vector3().fromArray(accessor.max)).applyMatrix4(matrix);box.max.y=Math.max(box.max.y,buildings.get(id).center[2]+buildings.get(id).height);
   if(footprints.has(id))footprints.get(id).union(box);else footprints.set(id,box);
  }
 }
 for(const child of node.children??[])visit(child,matrix);
}
for(const i of base.scenes[base.scene??0].nodes)visit(i);
const cellFor=new Map();for(const cell of manifest.cells)for(const id of cell.stats?.buildingIds??[])cellFor.set(id,cell);
const pixel=p=>{const q=p.clone().project(camera);return [(q.x+1)*960,(1-q.y)*540,q.z];};
function projectBox(box){const xs=[],ys=[];for(let i=0;i<8;i++){const p=pixel(new THREE.Vector3(i&1?box.max.x:box.min.x,i&2?box.max.y:box.min.y,i&4?box.max.z:box.min.z));xs.push(p[0]);ys.push(p[1]);}return [Math.min(...xs),Math.min(...ys),Math.max(...xs),Math.max(...ys)];}
const targetROI=[760,390,900,505],candidates=[];
for(const [id,box] of footprints){if(!frustum.intersectsBox(box))continue;const screen=projectBox(box),area=Math.max(0,Math.min(screen[2],targetROI[2])-Math.max(screen[0],targetROI[0]))*Math.max(0,Math.min(screen[3],targetROI[3])-Math.max(screen[1],targetROI[1]));if(area>0)candidates.push({id,name:buildings.get(id).name,cell:cellFor.get(id)?.id,rangeM:box.distanceToPoint(camera.position),screen,overlapROI:area});}
candidates.sort((a,b)=>b.overlapROI-a.overlapROI);

// Pick the farther wall occupying the supplied screenshot's white-building region.
// This is QA identification only, never a runtime location-specific rule.
const target=candidates.find(c=>c.rangeM>100),targetCell=manifest.cells.find(c=>c.id===target.cell);
assert.ok(target&&targetCell);
let lowest=Infinity,highest=-Infinity;for(const h of world.ground){lowest=Math.min(lowest,h);highest=Math.max(highest,h);}for(const b of world.buildings)highest=Math.max(highest,b.center[2]+b.height);
const demand=cell=>{let largest=0;const original=manifest.cells.find(c=>c.id===cell.id);for(const id of original?.stats?.buildingIds??[]){const box=footprints.get(id);if(box&&frustum.intersectsBox(box))largest=Math.max(largest,frontageScreenFraction(box,clip,camera.near));}return Math.sqrt(largest);};
const visibility=b=>{const box=new THREE.Box3(new THREE.Vector3(b[0],lowest,-b[3]),new THREE.Vector3(b[2],highest,-b[1]));return box.distanceToPoint(camera.position)<40||frustum.intersectsBox(box);};
const scene=new THREE.Scene(),loaded=[];
const controller=createFacadeMiddleController(middleIndex,{scene,yieldTask:async()=>{},prepare:async()=>{},
 load:async cell=>{const compressed=await readFile(path.join(publicDir,cell.url));const {createHash}=await import('node:crypto');assert.equal(createHash('sha256').update(compressed).digest('hex'),cell.sha256);loaded.push(cell.id);return gunzipSync(compressed);}});
controller.update({east:camera.position.x,north:-camera.position.z,forwardE:direction.x,forwardN:-direction.z,detailed:new Set(baseline.scope.active),screenDemand:demand,visible:visibility});
const initialStatus=controller.status();await controller.waitForIdle();const selected=controller.status();

assert.ok(selected.active.includes(target.cell),'Visible distant target cell did not receive actual production middle geometry');
assert.equal(selected.lastError,'');
// Decode only this real full-detail source cell, retaining triangles from the same facade/building layers as the packager.
const sourceBytes=Buffer.concat(await Promise.all(targetCell.chunks.map(async c=>gunzipSync(await readFile(path.join(publicDir,c))))));
const sourceLength=sourceBytes.readUInt32LE(12),doc=JSON.parse(sourceBytes.subarray(20,20+sourceLength).toString()),body=28+sourceLength;
const constructors={5126:Float32Array,5125:Uint32Array,5123:Uint16Array,5121:Uint8Array},widths={SCALAR:1,VEC2:2,VEC3:3,VEC4:4};
const array=i=>{const a=doc.accessors[i],v=doc.bufferViews[a.bufferView],C=constructors[a.componentType];return new C(sourceBytes.buffer,sourceBytes.byteOffset+body+(v.byteOffset??0)+(a.byteOffset??0),a.count*widths[a.type]);};
const sourceParts=[],targetMeshes=[],glassPoints=[];
for(const node of doc.nodes){if(node.mesh===undefined)continue;const mesh=doc.meshes[node.mesh],layer=node.extras?.layer??mesh.extras?.layer??node.name.split('__')[0],id=node.extras?.featureId??mesh.extras?.featureId??node.name.split('___')[0].split('__')[1];
 if(!id||!['buildings','facades','facade'].includes(layer))continue;
 const matrix=new THREE.Matrix4().fromArray(node.matrix??[1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1]);
 for(const primitive of mesh.primitives){const input=array(primitive.attributes.POSITION),position=new Float32Array(input.length),v=new THREE.Vector3();for(let i=0;i<input.length;i+=3){v.fromArray(input,i).applyMatrix4(matrix);v.toArray(position,i);}
  const indices=primitive.indices!==undefined?array(primitive.indices):Uint32Array.from({length:position.length/3},(_,i)=>i),material=doc.materials[primitive.material];
  const glazing=/glass|glazing/.test(((material.name??'')+' '+(node.extras?.material??'')).toLowerCase());
  sourceParts.push({position,indices,glazing,id});if(glazing)for(let i=0;i<position.length;i+=3)glassPoints.push({position:[position[i],position[i+1],position[i+2]],id});
  const geometry=new THREE.BufferGeometry().setAttribute('position',new THREE.BufferAttribute(position,3));geometry.setIndex(new THREE.BufferAttribute(indices,1));const object=new THREE.Mesh(geometry,new THREE.MeshBasicMaterial({side:THREE.DoubleSide}));object.userData={featureId:id,glazing};object.updateMatrixWorld();targetMeshes.push(object);
 }
}
const middleGroup=controller.groupFor(target.cell),middleParts=[];middleGroup.traverse(o=>{if(o instanceof THREE.Mesh)middleParts.push({position:o.geometry.attributes.position.array,indices:o.geometry.index.array,glazing:o.name.endsWith('__glazing')});});
const raycaster=new THREE.Raycaster(),rayHits=[];
for(const [x,y] of [[820,420],[850,445],[880,475]]){raycaster.setFromCamera(new THREE.Vector2(x/960-1,1-y/540),camera);const hit=raycaster.intersectObjects(targetMeshes,false)[0];rayHits.push({pixel:[x,y],featureId:hit?.object.userData.featureId,glazing:hit?.object.userData.glazing,distanceM:hit?.distance});}
assert.ok(rayHits.some(h=>h.featureId===target.id),'Selected screenshot wall does not intersect target real source triangles');
// Native-pixel geometry-only depth raster. White solid, dark glazing, transparent background; no image texture/lighting.
const crop=[740,385,920,515],W=crop[2]-crop[0],H=crop[3]-crop[1];
function raster(parts){const depth=new Float64Array(W*H).fill(Infinity),mask=new Uint8Array(W*H);let projectedTriangles=0,glazingTriangles=0;
 for(const part of parts){const projected=[];for(let i=0;i<part.position.length;i+=3)projected.push(pixel(new THREE.Vector3().fromArray(part.position,i)));
  for(let i=0;i<part.indices.length;i+=3){const p=projected[part.indices[i]],q=projected[part.indices[i+1]],r=projected[part.indices[i+2]];
   if(p[2]<-1||q[2]<-1||r[2]<-1||p[2]>1||q[2]>1||r[2]>1)continue;
   const x0=Math.max(crop[0],Math.floor(Math.min(p[0],q[0],r[0]))),x1=Math.min(crop[2]-1,Math.ceil(Math.max(p[0],q[0],r[0]))),y0=Math.max(crop[1],Math.floor(Math.min(p[1],q[1],r[1]))),y1=Math.min(crop[3]-1,Math.ceil(Math.max(p[1],q[1],r[1])));
   if(x1<x0||y1<y0)continue;const den=(q[1]-r[1])*(p[0]-r[0])+(r[0]-q[0])*(p[1]-r[1]);if(Math.abs(den)<1e-10)continue;
   projectedTriangles++;if(part.glazing)glazingTriangles++;
   for(let y=y0;y<=y1;y++)for(let x=x0;x<=x1;x++){const a=((q[1]-r[1])*(x+.5-r[0])+(r[0]-q[0])*(y+.5-r[1]))/den,b=((r[1]-p[1])*(x+.5-r[0])+(p[0]-r[0])*(y+.5-r[1]))/den,c=1-a-b;if(a<0||b<0||c<0)continue;const z=a*p[2]+b*q[2]+c*r[2],index=(y-crop[1])*W+x-crop[0];if(z<depth[index]){depth[index]=z;mask[index]=part.glazing?2:1;}}
  }
 }
 return {mask,projectedTriangles,glazingTriangles};
}
const sourceRaster=raster(sourceParts),middleRaster=raster(middleParts);
const compare=(label,sourceMask=sourceRaster.mask,middleMask=middleRaster.mask)=>{let source=0,middle=0,intersection=0,union=0;for(let i=0;i<W*H;i++){const a=label?sourceMask[i]===label:sourceMask[i]>0,b=label?middleMask[i]===label:middleMask[i]>0;source+=a;middle+=b;intersection+=a&&b;union+=a||b;}return {sourcePixels:source,middlePixels:middle,intersection,union,iou:union?intersection/union:1};};
const sourceGlassOnly=raster(sourceParts.filter(p=>p.glazing)),middleGlassOnly=raster(middleParts.filter(p=>p.glazing));
let different=0;for(let i=0;i<W*H;i++)different+=sourceRaster.mask[i]!==middleRaster.mask[i];
const bins=new Map(),key=p=>p.map(v=>Math.floor(v/.01)).join(',');for(const p of glassPoints){const k=key(p.position),list=bins.get(k)??[];list.push(p);bins.set(k,list);}
let matched=0,maxMeters=0,maxPixels=0,sumPixels=0;const matchedOwners=new Map();
for(const part of middleParts.filter(p=>p.glazing)){const owners=[];matchedOwners.set(part,owners);for(let i=0;i<part.position.length;i+=3){const p=Array.from(part.position.subarray(i,i+3)),cell=p.map(v=>Math.floor(v/.01));let closest,best=Infinity;
 for(let x=-1;x<=1;x++)for(let y=-1;y<=1;y++)for(let z=-1;z<=1;z++)for(const q of bins.get([cell[0]+x,cell[1]+y,cell[2]+z].join(','))??[]){const d=Math.hypot(...p.map((v,j)=>v-q.position[j]));if(d<best){best=d;closest=q;}}
 assert.ok(closest&&best<.002,'Middle glazing vertex did not preserve a source glazing position');owners.push(closest.id);const a=pixel(new THREE.Vector3().fromArray(p)),b=pixel(new THREE.Vector3().fromArray(closest.position)),delta=Math.hypot(a[0]-b[0],a[1]-b[1]);matched++;maxMeters=Math.max(maxMeters,best);maxPixels=Math.max(maxPixels,delta);sumPixels+=delta;
}}
const targetSourceGlass=sourceParts.filter(p=>p.glazing&&p.id===target.id),targetMiddleGlass=[];
for(const part of middleParts.filter(p=>p.glazing)){const owners=matchedOwners.get(part),indices=[];for(let i=0;i<part.indices.length;i+=3)if([0,1,2].every(k=>owners[part.indices[i+k]]===target.id))indices.push(...part.indices.subarray(i,i+3));if(indices.length)targetMiddleGlass.push({...part,indices});}
const targetSourceGlassTriangles=targetSourceGlass.reduce((n,p)=>n+p.indices.length/3,0),targetMiddleGlassTriangles=targetMiddleGlass.reduce((n,p)=>n+p.indices.length/3,0);assert.equal(targetSourceGlassTriangles,targetMiddleGlassTriangles);
const targetGlassProjection=compare(2,raster(targetSourceGlass).mask,raster(targetMiddleGlass).mask);
const actualGlazing=sourceParts.filter(p=>p.glazing).reduce((n,p)=>n+p.indices.length/3,0),packedGlazing=middleParts.filter(p=>p.glazing).reduce((n,p)=>n+p.indices.length/3,0);assert.equal(actualGlazing,packedGlazing);
const baseFacadeNodes=base.nodes.filter(n=>n.extras?.streamCell===target.cell&&n.extras?.class==='distant_exterior'&&n.extras?.layer==='facades');
assert.ok(baseFacadeNodes.length>0);const settledFallback=coverageRange(drawTier.baseFacade,0,1,0,false),settledMiddle=coverageRange(drawTier.facade,0,1,0,false);assert.deepEqual(settledFallback,[1,1]);assert.deepEqual(settledMiddle,[0,1]);
const report={kind:'CPU-only real production middle-controller selection plus real source/middle geometry projection; no browser/GPU/FPS or beauty-render validation.',camera:{...pose,viewport:[1920,1080],horizontalFov:THREE.MathUtils.radToDeg(2*Math.atan(Math.tan(THREE.MathUtils.degToRad(pose.fov)/2)*1920/1080))},target:{...target,identification:'Farther real building overlapping white-wall screenshot ROI, verified by rays against its actual source triangles.',building:buildings.get(target.id),sourceCell:targetCell.id,sourceCellHash:targetCell.contentHash,ownerScreenFraction:frontageScreenFraction(footprints.get(target.id),clip,camera.near),cellProjectedDemand:demand({id:target.cell}),rayHits},selection:{initialWanted:initialStatus.wanted,settledActive:selected.active,status:selected,targetReceivedMiddle:!!middleGroup,targetSelectionRank:initialStatus.wanted.indexOf(target.cell)+1,sourceDetailedCells:baseline.scope.active,baseFacadeFallback:{nodes:baseFacadeNodes.length,triangles:baseFacadeNodes.reduce((sum,n)=>sum+base.meshes[n.mesh].primitives.reduce((s,p)=>s+base.accessors[p.indices].count/3,0),0),metadataSelectsTier:'baseFacade',settledCoverageWithMiddle:0,settledMiddleCoverage:1,condition:'Production transition target reached after the 280ms facade ramp; verified with current coverageRange.'}},geometry:{sourceFacadeTriangles:sourceParts.reduce((n,p)=>n+p.indices.length/3,0),middleFacadeTriangles:middleParts.reduce((n,p)=>n+p.indices.length/3,0),sourceGlazingTriangles:actualGlazing,middleGlazingTriangles:packedGlazing,targetSourceGlazingTriangles:targetSourceGlassTriangles,targetMiddleGlazingTriangles:targetMiddleGlassTriangles,matchedGlazingVertices:matched,maxGlazingPositionErrorM:maxMeters,maxGlazingProjectionErrorPx:maxPixels,meanGlazingProjectionErrorPx:sumPixels/matched},projection:{cropXYXY:crop,nativeResolution:[1920,1080],sourceProjectedTriangles:sourceRaster.projectedTriangles,middleProjectedTriangles:middleRaster.projectedTriangles,allGeometry:compare(0),glazingWithSolidOcclusion:compare(2),glazingGeometryAlone:compare(2,sourceGlassOnly.mask,middleGlassOnly.mask),targetGlazingGeometryAlone:targetGlassProjection,differentClassPixels:different,classAgreement:1-different/(W*H)},limits:'The software mask compares solid/glazing geometry without trees, parked cars, wires, material textures, lighting, antialiasing or post-processing. It proves selection and retained projected openings, not final screenshot clarity. Current package triangle identity is per cell; geometry includes all buildings in that real source cell.'};
await writeFile(path.join(out,'middle-wall-projection.json'),JSON.stringify(report,null,2));
await writeFile(path.join(out,'middle-wall-masks.json'),JSON.stringify({crop,width:W,height:H,source:[...sourceRaster.mask],middle:[...middleRaster.mask]}));
console.log(JSON.stringify({target:target.name,cell:target.cell,range:target.rangeM,selectionRank:report.selection.targetSelectionRank,geometry:report.geometry,projection:report.projection},null,2));await controller.dispose();for(const m of targetMeshes){m.geometry.dispose();m.material.dispose();}
