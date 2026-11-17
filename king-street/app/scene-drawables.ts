import * as THREE from 'three';
import {retainParserImageSources,releaseParserImageSources} from './scene-image-pool.ts';
import type {GLTF} from 'three/addons/loaders/GLTFLoader.js';
import type {World} from './king-pilot.ts';
import {finishDrawableParser,releaseSceneLoadingData} from './scene-stream.ts';
import {sceneCpuSlice} from './scene-flat.ts';
import {ensureGeometryNormals,ensureGeometryNormalsAsync,packStaticNormals,finishStaticSteps,finishStaticStepsAsync} from './static-detail-geometry.ts';
import {batchPaintSwatches,batchStaticWorld,batchStaticWorldAsync} from './static-detail.ts';
import {finishMaterials,classOf} from './finish.ts';
import {weatherMaterials,surfaceKind,hasObservedPattern,WEATHERED_LAYERS,UNWEATHERED} from './weathering.ts';
import {glazeMaterials,isGlazing,paneAttributes,paneAttributesAsync,type GlazedWindow,type GlazingEmitter} from './glazing.ts';
import {mergeTintVariants} from './material-merge.ts';
import {retireMaterialResources,disposeOwnedTextures} from './material-resources.ts';
import {releaseTextureCpuImages} from './compressed-textures.ts';
import {detailRoadPaint,detailAsphalt,detailGrass,detailGarden,detailPavement,ASPHALT_CLASS,GRASS_CLASS} from './ground-detail.ts';
import {buildFoliage,finishFoliage,barkMaterials,taperTreeLimbs} from './foliage.ts';
import {finishCanopyImpostor} from './canopy-impostor.ts';
import {NO_SUN_SHADOW,THIN_SHADOW_DETAIL,smallShadowDetail} from './shadow-policy.ts';
import {collisionOnly} from './collision-only.ts';
import {batchNeighborhoodSubmissions} from './neighborhood-submissions.ts';

export type SceneDrawableShared = {
 data: Pick<World,'grid'|'ground'>;
 renderer: {capabilities: Pick<THREE.WebGLRenderer['capabilities'],'getMaxAnisotropy'>;
   extensions?: {has(name:string):boolean}};
 atmosphere: {setupMaterial(material: THREE.Material|THREE.Material[]): void};
 leafTexture: THREE.Texture;
 groundAt(this:void,x:number,z:number): number;
 wxGround: THREE.DataTexture;
 roadField: Record<string,THREE.IUniform>;
 paving: Record<string,THREE.IUniform>;
 roadDamp: THREE.IUniform<number>;
 roadReflection: THREE.IUniform<THREE.Texture|null>;
 surfaceNight: THREE.IUniform<number>;
};
export type SceneDrawableOptions = {
 prepareRenderable?: (mesh:THREE.Mesh)=>void;
 sharedFoliage?: boolean;
 sceneHash?: string;
 /** Caller prefixes these local stage names with the cell's own identifier. */
 stage?: (name:string)=>void;
 /** Streaming can yield at phase boundaries; startup keeps its original schedule. */
 yieldCpu?: ()=>Promise<void>;
 signal?: AbortSignal;
};
function getLayer(o:THREE.Object3D){return String(o.userData.layer||o.name.split('___')[0].split('__')[0]);}
function getId(o:THREE.Object3D){return String(o.userData.featureId||o.name.split('___')[0].split('__')[1]||'');}
function materialTextures(materials:Iterable<THREE.Material>, excluded:ReadonlySet<THREE.Texture>) {
 const textures=new Set<THREE.Texture>();
 for(const material of materials)for(const [property,value] of Object.entries(material))
  if(property!=='envMap'&&value instanceof THREE.Texture&&!value.isRenderTargetTexture&&!excluded.has(value))textures.add(value);
 return textures;
}

/** The startup and streamed-detail paths use exactly the same finishes, shader
 * hooks, tint merging, leaf cards and static batches. The identity root is kept
 * unattached until callers have finished staging uploads; exported transforms
 * remain absolute world coordinates, including the model's z-up conversion.
 * Each parsed model owns its materials and geometry. Shared uniforms and the
 * leaf/environment textures remain owned by the viewer. */
export async function prepareSceneDrawables(model:GLTF,shared:SceneDrawableShared,options:SceneDrawableOptions={}) {
 const {data,renderer,leafTexture,groundAt,wxGround,roadField,paving,roadDamp,roadReflection,surfaceNight}=shared;
 const loadedHash=options.sceneHash;
 const root=new THREE.Group();root.name='scene-drawables';
 const sharedTextures=new Set<THREE.Texture>([leafTexture,wxGround]);
 for(const uniforms of [roadField,paving])for(const uniform of Object.values(uniforms))if(uniform.value instanceof THREE.Texture)sharedTextures.add(uniform.value);
 if(roadReflection.value)sharedTextures.add(roadReflection.value);
 const materials=new Set<THREE.Material>(),textures=new Set<THREE.Texture>(),geometries=new Set<THREE.BufferGeometry>();
 // Remove retired resources as soon as the normal merge pipeline disposes them,
 // so final cleanup is once-only even for aliases and partially prepared cells.
 const trackMaterial=(material:THREE.Material|THREE.Material[])=>{
  for(const m of Array.isArray(material)?material:[material]){
   if(materials.has(m))continue;materials.add(m);
   const retire=()=>{materials.delete(m);m.removeEventListener('dispose',retire);};m.addEventListener('dispose',retire);
   for(const texture of materialTextures([m],sharedTextures))if(!textures.has(texture)){
    textures.add(texture);const retireTexture=()=>{textures.delete(texture);texture.removeEventListener('dispose',retireTexture);};texture.addEventListener('dispose',retireTexture);
   }
  }
 };
 const collect=()=>root.traverse(object=>{if(object instanceof THREE.Mesh){geometries.add(object.geometry);trackMaterial(object.material);}});
 const atmosphere={setupMaterial:(material:THREE.Material|THREE.Material[])=>{trackMaterial(material);shared.atmosphere.setupMaterial(material);}};
 let disposed=false;
 let submissions:ReturnType<typeof batchNeighborhoodSubmissions>|undefined;
 const dispose=()=>{
  if(disposed)return;disposed=true;root.removeFromParent();
  if(submissions){for(const mesh of submissions.meshes)geometries.delete(mesh.geometry);submissions.dispose();}
  disposeOwnedTextures(textures);textures.clear();releaseParserImageSources(model.parser);
  for(const geometry of geometries)geometry.dispose();geometries.clear();
  for(const material of materials)material.dispose();materials.clear();
  model.parser.associations.clear();root.clear();
 };
 const glazingEmitters:GlazingEmitter[]=[];
 const yieldPreparation=options.yieldCpu ? async()=>{options.signal?.throwIfAborted();await options.yieldCpu!();options.signal?.throwIfAborted();}:undefined;
 const cpuSlice=sceneCpuSlice(yieldPreparation);
 const checkpoint=async(name:string)=>{options.stage?.(name);options.signal?.throwIfAborted();if(options.yieldCpu)await options.yieldCpu();options.signal?.throwIfAborted();};
 try {
 releaseSceneLoadingData(model.parser);
 root.add(model.scene);collect();
 options.signal?.throwIfAborted();
 const objects:THREE.Mesh[]=[];const originals=new Map<THREE.Mesh,THREE.Material|THREE.Material[]>();
 const sourceMeshes:THREE.Mesh[]=[];model.scene.traverse(o=>{if(o instanceof THREE.Mesh)sourceMeshes.push(o);});
 for(const o of sourceMeshes){
 const pause=cpuSlice();if(pause)await pause;
 if(collisionOnly(o)){objects.push(o);originals.set(o,o.material);o.visible=false;o.castShadow=false;o.receiveShadow=false;continue;}
 if(options.sharedFoliage&&getLayer(o)==='vegetation'&&o.userData.foliageFanCentres){o.visible=false;continue;}
 objects.push(o);o.castShadow=!['terrain','roads','markings','sidewalks','cycleways'].some(n=>o.name.startsWith(n));o.receiveShadow=true;
 // The terrain and the lawns draped on it come without normals, so they were drawn flat-shaded: a quilt of 3 m facets on
 // every slope. They are height fields: each vertex takes the slope of the world's ground grid under it, and its own
 // copy of the material is drawn smooth-shaded.
 if(['terrain','landscape'].includes(getLayer(o))&&!o.geometry.attributes.normal&&!Array.isArray(o.material)){
  const p=o.geometry.attributes.position,v=new THREE.Vector3(),w=new THREE.Vector3(),n=new Float32Array(p.count*3);o.updateMatrixWorld(true);
  const toLocal=new THREE.Matrix3().getNormalMatrix(o.matrixWorld).invert();   // the model's root turns z-up into y-up
  for(let i=0;i<p.count;i++){if(i%1024===0){const pause=cpuSlice();if(pause)await pause;}v.fromBufferAttribute(p,i).applyMatrix4(o.matrixWorld);const hx=(groundAt(v.x+1.5,v.z)-groundAt(v.x-1.5,v.z))/3,hz=(groundAt(v.x,v.z+1.5)-groundAt(v.x,v.z-1.5))/3;w.set(-hx,1,-hz).applyMatrix3(toLocal).normalize();n[i*3]=w.x;n[i*3+1]=w.y;n[i*3+2]=w.z;}
  o.geometry.setAttribute('normal',new THREE.BufferAttribute(n,3));
  const m=o.material.clone() as THREE.MeshStandardMaterial;m.flatShading=false;atmosphere.setupMaterial(m);o.material=m;originals.set(o,m);}
  // Meshes exported without normals are shaded flat from screen derivatives, but the shadow lookup still needs a
  // vertex normal for its bias: a zero normal makes that term undefined and every shadow test fails silently.
  if(!o.userData.foliageFanCentres){if(yieldPreparation)await ensureGeometryNormalsAsync(o.geometry,yieldPreparation);else ensureGeometryNormals(o.geometry);}
  if(o.material instanceof THREE.MeshStandardMaterial){for(const t of [o.material.map,o.material.normalMap,o.material.roughnessMap]){if(t){const anisotropy=Math.min(o.userData.canopyImpostor?2:8,renderer.capabilities.getMaxAnisotropy());if(t.anisotropy!==anisotropy){t.anisotropy=anisotropy;t.needsUpdate=true;}}}}
  originals.set(o,o.material);atmosphere.setupMaterial(o.material);
  if(o.userData.canopyImpostor && o.material instanceof THREE.MeshStandardMaterial)finishCanopyImpostor(o.material,leafTexture);
 }
 // The neighborhood's distant facades already carry the original surfaces'
 // combined albedo. They still receive sunlight, shadows and fine filtering.
 const sourceSurfaces=objects.filter(o=>!o.userData.bakedAppearance&&!collisionOnly(o));
 // A material shared by a building layer and the ground (one concrete scan for facades and pavements, one white for
 // trim and road paint) is copied for the ground meshes: the depth priority below is set per material, and a pavement
 // that inherited a facade's pull was drawn over the road in front of it. The ground stacks by its real heights (road
 // over terrain, paint over road, kerb over road) and takes no priority.
 {const ranked=new Set(['facades','signs','roofs','buildings']);const shared=new Set<THREE.Material>();const copies=new Map<THREE.Material,THREE.Material>();
  for(const o of objects)if(ranked.has(getLayer(o)))for(const m of Array.isArray(o.material)?o.material:[o.material])shared.add(m);
  const own=(m:THREE.Material)=>{if(!shared.has(m))return m;let c=copies.get(m);if(!c){c=m.clone();atmosphere.setupMaterial(c);copies.set(m,c);}return c;};
  for(const o of objects)if(!ranked.has(getLayer(o))){o.material=Array.isArray(o.material)?o.material.map(own):own(o.material);originals.set(o,o.material);}}
 // Each material class gets the finish of the surface it names; glazing, metal, paint and asphalt reflect a local probe.
 await checkpoint('sources');
 const probeMaterials=finishMaterials(sourceSurfaces);
 // Window and door reveals are drawn by the generator as a near-black 'recess' surface, which outlined every opening in thick
 // black; a reveal is the wall turned inward, so each takes its building's main wall material (the occlusion passes shade it
 // as a recess). Reveals are many thin strips; a recess mesh of large panels (a venue's dark backings) keeps its finish.
 {const wallClass=/^(brick|render|stucco|stone|limestone|cast_stone|concrete|terracotta|timber|panel_dark|venue_concrete)/;const area=new Map<string,Map<THREE.Material,number>>();
  for(const o of objects){const l=getLayer(o);if(l!=='buildings'&&l!=='facades')continue;if(!wallClass.test(classOf(o.name))||Array.isArray(o.material))continue;const id=getId(o);if(!id)continue;
   const g=o.geometry,n=(g.index?g.index.count:g.attributes.position.count)/3;let m=area.get(id);if(!m){m=new Map();area.set(id,m);}m.set(o.material,(m.get(o.material)??0)+n);}
  const thin=async(o:THREE.Mesh)=>{const g=o.geometry,p=g.attributes.position,idx=g.index,n=idx?idx.count:p.count,A:number[]=[],a=new THREE.Vector3(),b=new THREE.Vector3(),c=new THREE.Vector3();
   for(let i=0;i<n;i+=3){if(i%3072===0){const pause=cpuSlice();if(pause)await pause;}a.fromBufferAttribute(p,idx?idx.getX(i):i);b.fromBufferAttribute(p,idx?idx.getX(i+1):i+1);c.fromBufferAttribute(p,idx?idx.getX(i+2):i+2);A.push(b.sub(a).cross(c.sub(a)).length()/2);}
   A.sort((x,y)=>x-y);return A.length>0&&A[A.length>>1]<.25&&A[A.length-1]<2;};
  for(const o of objects){const pause=cpuSlice();if(pause)await pause;if(o.userData.startupWallReveal||classOf(o.name)!=='recess'||Array.isArray(o.material)||!await thin(o))continue;const m=area.get(getId(o));if(!m)continue;let best:THREE.Material|null=null,bn=-1;for(const [mat,n] of m)if(n>bn){best=mat;bn=n;}if(best){o.material=best;originals.set(o,best);}}}
 // One material per surface class where the buildings' own copies differ only by a tint (material-merge.ts): the tint
 // goes into a vertex colour, the copies' maps are never uploaded, and the class batches across buildings below.
 {const built=sourceSurfaces.filter(o=>['buildings','facades','foundations','roofs'].includes(getLayer(o)));
  // the depth priority's layer rank (below): merged only within one rank
  const rankOf=(o:THREE.Mesh)=>{const l=getLayer(o);return l==='facades'||l==='signs'?3:l==='roofs'?2:l==='buildings'?(o.name.includes('__observed_')?1:0):-1;};
  const cpuMaterials=[...new Set([...objects.flatMap(o=>Array.isArray(o.material)?o.material:[o.material]),
   ...[...model.parser.associations.keys()].filter((item):item is THREE.Material=>item instanceof THREE.Material)])];
  try {
  const {replaced}=await mergeTintVariants(built,o=>classOf(o.name),rankOf,loadedHash ? {
   sceneHash:loadedHash, textureIndex:t=>model.parser.associations.get(t)?.textures,
  } : undefined, yieldPreparation);
  const active=objects.flatMap(o=>Array.isArray(o.material)?o.material:[o.material]);
  // Earlier surface/reveal substitutions can orphan loaded materials too.
  // Include parser associations so their decoded images and texture promises
  // do not survive merely because they were not replaced by the tint merge.
  const kept=new Set(active),retired=new Set<THREE.Material>(replaced);
  for(const item of model.parser.associations.keys())if(item instanceof THREE.Material&&!kept.has(item))retired.add(item);
  retireMaterialResources([...retired],active,model.parser.associations);
  retainParserImageSources(model.parser,active);
  for(const o of objects)originals.set(o,o.material);
  }finally{releaseTextureCpuImages(cpuMaterials,model.parser);}}
 // Street grime, rain streaks and tone drift on walls, patched membranes on flat roofs (world space, no per-building data).
 await checkpoint('materials');
 // the road field's uniforms (roadfield.ts): its texture is drawn once the world is batched, before the first frame
 let weathering:ReturnType<typeof weatherMaterials>|null=null;let glazedWindows:GlazedWindow[]=[];
 {const tex=wxGround;const wx=new Set<THREE.MeshStandardMaterial>();
  for(const o of sourceSurfaces){if(!WEATHERED_LAYERS.has(getLayer(o))||UNWEATHERED.test(classOf(o.name)))continue;for(const m of Array.isArray(o.material)?o.material:[o.material])if(m instanceof THREE.MeshStandardMaterial)wx.add(m);}
  // Untyped surfaces may receive their class's procedural character. Typed
  // observed patterns keep their own scan/normal map; weathering still adds
  // world-space grime and mottle, but must not lay a second default joint grid.
  const kinds=new Map<THREE.Material,number>(),observedPatterns=new Set<THREE.Material>();
  for(const o of sourceSurfaces){const k=surfaceKind(classOf(o.name)),observed=hasObservedPattern(o.name);for(const m of Array.isArray(o.material)?o.material:[o.material]){
   if(observed)observedPatterns.add(m);else if(k)kinds.set(m,k);
  }}
  for(const m of observedPatterns)kinds.delete(m);
  weathering=weatherMaterials(wx,tex,data.grid,kinds);
  // A dim room behind every window instead of an opaque black pane.
  const glass=new Set<THREE.MeshStandardMaterial>(),glassMeshes:THREE.Mesh[]=[],glassCls=new Map<THREE.Material,string>();
  for(const o of sourceSurfaces){const c=classOf(o.name);if(!isGlazing(c))continue;glassMeshes.push(o);for(const m of Array.isArray(o.material)?o.material:[o.material])if(m instanceof THREE.MeshStandardMaterial){glass.add(m);glassCls.set(m,c);}}
  // every window its own coordinates and identity, so each gets its own blinds, curtains or room (glazing.ts)
  glazedWindows=yieldPreparation?await paneAttributesAsync(glassMeshes,glazingEmitters,glassCls,yieldPreparation):paneAttributes(glassMeshes,glazingEmitters,glassCls);glazeMaterials(glass,glassCls,surfaceNight);
  // Worn asphalt and grown lawns instead of one repeating texture and one flat green (ground-detail.ts).
  const ofClass=(re:RegExp)=>{const out=new Set<THREE.MeshStandardMaterial>();for(const o of sourceSurfaces){if(!re.test(classOf(o.name)))continue;for(const m of Array.isArray(o.material)?o.material:[o.material])if(m instanceof THREE.MeshStandardMaterial)out.add(m);}return out;};
  detailRoadPaint(sourceSurfaces.filter(o=>getLayer(o)==='markings'),originals,atmosphere.setupMaterial);
  detailAsphalt(ofClass(ASPHALT_CLASS),{...roadField,uRoadDamp:roadDamp,uRoadReflection:roadReflection});barkMaterials(ofClass(/^bark$/));detailGrass(ofClass(GRASS_CLASS));
  // the bare terrain between buildings is mostly back gardens, not one plain of paving
  const terrain=new Set<THREE.MeshStandardMaterial>();for(const o of sourceSurfaces)if(getLayer(o)==='terrain'&&o.material instanceof THREE.MeshStandardMaterial)terrain.add(o.material);detailGarden(terrain,roadField);
  // sidewalk concrete (not its joints or tactile strips), on its own copy of a material other concrete shares
  const walks=new Map<THREE.Material,THREE.MeshStandardMaterial>();
  for(const o of sourceSurfaces)if(getLayer(o)==='sidewalks'&&classOf(o.name)==='concrete'&&o.material instanceof THREE.MeshStandardMaterial){let m=walks.get(o.material);if(!m){m=o.material.clone();atmosphere.setupMaterial(m);walks.set(o.material,m);}o.material=m;originals.set(o,m);}
  detailPavement(walks.values(),{...roadField,...paving});}
 await checkpoint('weathering');
 // Coincident surfaces (a tier wall on the outline over the outer wall's top band, glazing in its curtain panel's
 // plane, an observed skin on its host wall) z-fight: the winner flips per pixel as the camera moves. Each material
 // gets a fixed depth priority instead: facade detail over roof caps over observed skins over host massing, with a
 // stable tie-break between materials of one layer. Use only fixed depth-buffer units:
 // a slope-dependent offset becomes enormous on grazing return faces and pulls
 // buried bay seams over windows and reliefs. Geometry keeps its actual depth.
 {const layerRank:Record<string,number>={facades:3,signs:3,roofs:2,buildings:0};const rank=new Map<THREE.Material,number>();
  for(const o of objects){const l=getLayer(o);let r=layerRank[l];if(r===undefined)continue;if(l==='buildings'&&o.name.includes('__observed_'))r=1;
   for(const m of Array.isArray(o.material)?o.material:[o.material])rank.set(m,Math.max(rank.get(m)??-1,r));}
  const byName=[...rank.keys()].sort((a,b)=>a.name.localeCompare(b.name));
  byName.forEach((m,i)=>{const r=rank.get(m)!*3+(i%3);if(r>0){m.polygonOffset=true;m.polygonOffsetFactor=0;m.polygonOffsetUnits=-2*r;m.needsUpdate=true;}});}
 for(const o of objects){const pause=cpuSlice();if(pause)await pause;if(getLayer(o)==='vegetation'&&classOf(o.name)==='bark')taperTreeLimbs(o);}
 const normalPacking=packStaticNormals(new Set(objects.map(o=>o.geometry)));
 if(yieldPreparation)await finishStaticStepsAsync(normalPacking,yieldPreparation);else finishStaticSteps(normalPacking);
 // Meshes with the same finish share a draw. Heavy batches keep local bounds for
 // frustum culling, while picking, evidence and collision retain the sources.
 batchPaintSwatches(sourceSurfaces,getLayer,originals);
 const batchRoot=new THREE.Group();batchRoot.name='batches';root.add(batchRoot);
 const shadowClass=(o:THREE.Mesh)=>{const cls=classOf(o.name);return [NO_SUN_SHADOW.test(cls),smallShadowDetail(cls,getLayer(o)),THIN_SHADOW_DETAIL.test(cls),o.userData.middleOwner??o.userData.streamCell??'',o.userData.class==='collision_hull'?'collision':''].join('|');};
 const batches=yieldPreparation?await batchStaticWorldAsync(objects,getLayer,shadowClass,batchRoot,yieldPreparation):batchStaticWorld(objects,getLayer,shadowClass,batchRoot);
 await checkpoint('batches');
 // Dense leaf-cluster canopies grown on the pipeline's leaf quads (which are hidden; the layer toggle handles both).
 {const leaves=objects.filter(o=>getLayer(o)==='vegetation'&&classOf(o.name).startsWith('leaf')&&!o.userData.batched);
  if(leaves.length){const foliage=buildFoliage(leaves,leafTexture);atmosphere.setupMaterial(foliage.material);finishFoliage(foliage.material);
   for(const mesh of foliage.meshes){root.add(mesh);batches.push(mesh);}
   for(const o of leaves){o.visible=false;o.userData.batched=foliage.meshes[0];}}}
 // The occlusion pass draws depth and normals without alpha: every leaf card would be a solid square casting dark
 // halos through the crowns, so the canopies are left out of that pass (they still receive the result).
 const renderables=[...batches,...objects.filter(o=>!o.userData.batched&&!collisionOnly(o))];
 const foliageBatches=renderables.filter(mesh=>mesh.name==='vegetation___foliage'||
  Boolean(mesh.userData.canopyImpostor||mesh.userData.members?.[0]?.userData.canopyImpostor));
 if(options.prepareRenderable)for(const mesh of renderables)options.prepareRenderable(mesh);
 submissions=batchNeighborhoodSubmissions(renderables,root,{
  supported:shared.renderer.extensions?.has('WEBGL_multi_draw')===true,layerOf:getLayer,
 });
 for(const controller of submissions.controllers)geometries.add(controller.geometry);
 await checkpoint('ready');
 collect();
 finishDrawableParser(model.parser);
 return {model,root,objects,originals,renderables,submissionMeshes:submissions.meshes,submissions,probeMaterials,batchRoot,batches,foliageBatches,glazedWindows,glazingEmitters,weathering,textures,
  get disposed(){return disposed;},dispose,
 };
 }catch(error){collect();dispose();throw error;}

}
export type SceneDrawables = Awaited<ReturnType<typeof prepareSceneDrawables>>;

/** Explicit small upload steps: callers run one step per animation frame while
 * the root is still unattached, then attach after pending reaches zero. */
export function createDrawableTextureUpload(drawables:SceneDrawables,renderer:Pick<THREE.WebGLRenderer,'initTexture'>){
 const pending=[...drawables.textures];let cursor=0;
 return {
  get pending(){return drawables.disposed?0:pending.length-cursor;},
  uploadNext(maxTextures=1){
   if(drawables.disposed)return 0;
   const limit=Math.max(0,Math.floor(maxTextures));let count=0;
   while(cursor<pending.length&&count<limit){renderer.initTexture(pending[cursor]);cursor++;count++;}
   return count;
  },
 };
}
