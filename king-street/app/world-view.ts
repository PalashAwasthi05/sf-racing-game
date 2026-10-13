import * as THREE from 'three';
import {GLTFLoader} from 'three/addons/loaders/GLTFLoader.js';
import {mergeGeometries} from 'three/addons/utils/BufferGeometryUtils.js';
import {OrbitControls} from 'three/addons/controls/OrbitControls.js';
import {EffectComposer} from 'three/addons/postprocessing/EffectComposer.js';
import {GTAOPass} from 'three/addons/postprocessing/GTAOPass.js';
import type {World} from './king-pilot';
import {createCar,buildDistanceField,loadRoadster,type Car,type CarInput} from './car';
import {buildCarGround,Obstacles} from './car-ground';
import {loadParkedCars} from './parked-cars';
import {loadOverhead} from './overhead';
import {createAtmosphere,PRESET_ORDER,type Preset} from './atmosphere';
import {finishMaterials,classOf} from './finish';
import {weatherMaterials,groundTexture,surfaceKind,WEATHERED_LAYERS,UNWEATHERED} from './weathering';
import {glazeMaterials,isGlazing,paneAttributes,type GlazedWindow} from './glazing';
import {buildFacadeMap} from './facademap';
import {mergeTintVariants} from './material-merge';
import {detailAsphalt,detailGrass,detailGarden,detailPavement,ASPHALT_CLASS,GRASS_CLASS} from './ground-detail';
import {buildFoliage,finishFoliage,barkMaterials} from './foliage';
import {createGrade} from './grade';
import {MultisampledRenderPass,buildDepthProxies,installDepthProxies,cullTransmission,cacheArrayUniforms} from './render-budget';
import {attachCarLod} from './car-lod';
import {roadSpots,spawnOnRoad,onRoad,type RoadSpots} from './car-spawn';
import {buildSkyVisibility,occludeMaterials} from './occlusion';
import {buildRoadField} from './roadfield';
import {buildPavingField} from './paving';
import {loadHorizon} from './horizon';

export async function createWorldView(host:HTMLDivElement,data:World,onSelect:(id:string)=>void,onProgress:(message:string)=>void,modelUrl?:string){
 const scene=new THREE.Scene();
 const roadster=loadRoadster();   // the car model downloads while the world loads
 // far enough for the backdrop (horizon.ts: the East Bay hills, Mount Tamalpais, Mount Diablo); depth precision near the
 // camera is set by the near plane, not the far one
 const camera=new THREE.PerspectiveCamera(58,1,.18,60000);
 // No canvas antialiasing: the canvas only receives the output pass's full-screen quad (the 3D image is antialiased in
 // its own multisampled target below), so canvas samples were memory and a resolve every frame for nothing.
 const renderer=new THREE.WebGLRenderer({antialias:false,powerPreference:'high-performance'});
 // Render scale adapts to the frame time between 0.7 and the device ratio (capped at 1.25); everything sized from it.
 // The most it renders is 1.25 render pixels per CSS pixel and four million pixels in all: a maximised window on a 5K
 // display at 1.25 was 5.3 million, and the frame's cost grows with them (antialiasing, occlusion, post-processing).
 const PIXEL_BUDGET=4e6;
 const scaleCap=()=>Math.max(.5,Math.min(window.devicePixelRatio,1.25,Math.sqrt(PIXEL_BUDGET/Math.max(1,host.clientWidth*host.clientHeight))));
 let maxScale=scaleCap();let renderScale=maxScale,qualityLocked=false;renderer.setPixelRatio(renderScale);renderer.shadowMap.enabled=true;renderer.shadowMap.type=THREE.PCFSoftShadowMap;renderer.toneMapping=THREE.ACESFilmicToneMapping;renderer.toneMappingExposure=.85;
 host.appendChild(renderer.domElement);renderer.domElement.setAttribute('aria-label','3D neighborhood. Use the camera controls to explore.');
 const controls=new OrbitControls(camera,renderer.domElement);controls.enableDamping=true;controls.dampingFactor=.08;controls.minDistance=4;controls.maxDistance=700;controls.maxPolarAngle=Math.PI*.49;
 // Sky, clouds, sun, sky-lit environment, fog, exposure and cascaded shadows live in one module with three presets.
 const atmosphere=createAtmosphere(scene,renderer,camera);
 // The lit scene is drawn into a multisampled target: the composer's default target has none, and the canvas's own
 // antialiasing never touches the 3D image once post-processing is on, so every edge and window frame was aliased.
 // Only the scene pass is multisampled; the post passes read its resolved copy on single-sample, depthless targets.
 const composer=new EffectComposer(renderer,new THREE.WebGLRenderTarget(1,1,{type:THREE.HalfFloatType,depthBuffer:false}));const scenePass=new MultisampledRenderPass(scene,camera);composer.addPass(scenePass);
 // Ambient occlusion at the scale of buildings (inner corners, under cornices and eaves, the foot of every wall). It draws
 // its own normals and depth at its own resolution: reading the scene pass's full-resolution depth instead was slower (the
 // samples scatter over a 20 MB texture) and would let the canopies' leaf cards occlude themselves.
 const ao=new GTAOPass(scene,camera,1,1);ao.updateGtaoMaterial({radius:2.5,distanceExponent:1.4,thickness:1.5,samples:8});ao.updatePdMaterial({samples:8});ao.blendIntensity=.9;
 ao.output=-1;ao.needsSwap=false;   // computed only: the composite applies it
 composer.addPass(ao);
 // ... applied to the indirect share of each pixel's light only, with the exposure, in one pass over the scene's own
 // resolved image (no copy of it); then the bloom and the grade, tone mapping and output in one more (grade.ts).
 scenePass.direct=true;
 const grade=createGrade(composer,scenePass.target.texture,ao);
 let aoScale=.4;
 function lighting(preset:string){const p=(PRESET_ORDER as string[]).includes(preset)?preset as Preset:'afternoon';atmosphere.apply(p);grade.preset(p);}
 let carsUrl:string|null=null,overheadUrl:string|null=null;   // the world's parked cars and overhead equipment, when its manifest names them
 async function loadBuffer(){
 if(modelUrl){const response=await fetch(modelUrl);if(!response.ok)throw new Error('Geometry preview unavailable');return response.arrayBuffer();}
 // `?manifest=/compare/x.json` loads another published build (local comparison only; same-origin paths only).
 const requested=new URL(window.location.href).searchParams.get('manifest');const manifestPath=requested&&/^\/compare\/[\w.-]+\.json$/.test(requested)?requested:'/model.json';
 const manifest=await fetch(manifestPath,{cache:'no-cache'}).then(r=>{if(!r.ok)throw new Error('Scene manifest unavailable');return r.json() as Promise<{chunks:string[];compressedBytes:number;cars?:string;overhead?:string}>});carsUrl=manifest.cars??null;overheadUrl=manifest.overhead??null;
 let loaded=0;
 const packed=new ReadableStream<BufferSource>({async start(controller){try{for(const path of manifest.chunks){const response=await fetch(path);if(!response.ok||!response.body)throw new Error('Scene download failed');const reader=response.body.getReader();while(true){const {done,value}=await reader.read();if(done)break;loaded+=value.byteLength;onProgress('Loading detailed scene… '+Math.round(loaded/manifest.compressedBytes*100)+'%');controller.enqueue(value)}}controller.close()}catch(e){controller.error(e)}}});
 return new Response(packed.pipeThrough(new DecompressionStream('gzip'))).arrayBuffer();
 }
 const buffer=await loadBuffer();onProgress('Preparing materials and geometry…');
 const downloadUrl=URL.createObjectURL(new Blob([buffer],{type:'model/gltf-binary'}));
 const model=await new GLTFLoader().parseAsync(buffer,'/');scene.add(model.scene);
 const objects:THREE.Mesh[]=[];const originals=new Map<THREE.Mesh,THREE.Material|THREE.Material[]>();const evidences=new Map<THREE.Mesh,THREE.Material>();
 model.scene.traverse(o=>{if(!(o instanceof THREE.Mesh))return;objects.push(o);o.castShadow=!['terrain','roads','markings','sidewalks','cycleways'].some(n=>o.name.startsWith(n));o.receiveShadow=true;
 // The terrain and the lawns draped on it come without normals, so they were drawn flat-shaded: a quilt of 3 m facets on
 // every slope. They are height fields: each vertex takes the slope of the world's ground grid under it, and its own
 // copy of the material is drawn smooth-shaded.
 for(const o of objects)if(['terrain','landscape'].includes(getLayer(o))&&!o.geometry.attributes.normal&&!Array.isArray(o.material)){
  const p=o.geometry.attributes.position,v=new THREE.Vector3(),w=new THREE.Vector3(),n=new Float32Array(p.count*3);o.updateMatrixWorld(true);
  const toLocal=new THREE.Matrix3().getNormalMatrix(o.matrixWorld).invert();   // the model's root turns z-up into y-up
  for(let i=0;i<p.count;i++){v.fromBufferAttribute(p,i).applyMatrix4(o.matrixWorld);const hx=(groundAt(v.x+1.5,v.z)-groundAt(v.x-1.5,v.z))/3,hz=(groundAt(v.x,v.z+1.5)-groundAt(v.x,v.z-1.5))/3;w.set(-hx,1,-hz).applyMatrix3(toLocal).normalize();n[i*3]=w.x;n[i*3+1]=w.y;n[i*3+2]=w.z;}
  o.geometry.setAttribute('normal',new THREE.BufferAttribute(n,3));
  const m=o.material.clone() as THREE.MeshStandardMaterial;m.flatShading=false;atmosphere.setupMaterial(m);o.material=m;originals.set(o,m);}
  // Meshes exported without normals are shaded flat from screen derivatives, but the shadow lookup still needs a
  // vertex normal for its bias: a zero normal makes that term undefined and every shadow test fails silently.
  if(!o.geometry.attributes.normal)o.geometry.computeVertexNormals();
  if(o.material instanceof THREE.MeshStandardMaterial){for(const t of [o.material.map,o.material.normalMap,o.material.roughnessMap]){if(t){t.anisotropy=Math.min(8,renderer.capabilities.getMaxAnisotropy());t.needsUpdate=true;}}}
  originals.set(o,o.material);atmosphere.setupMaterial(o.material);
 });
 // A material shared by a building layer and the ground (one concrete scan for facades and pavements, one white for
 // trim and road paint) is copied for the ground meshes: the depth priority below is set per material, and a pavement
 // that inherited a facade's pull was drawn over the road in front of it. The ground stacks by its real heights (road
 // over terrain, paint over road, kerb over road) and takes no priority.
 {const ranked=new Set(['facades','signs','roofs','buildings']);const shared=new Set<THREE.Material>();const copies=new Map<THREE.Material,THREE.Material>();
  for(const o of objects)if(ranked.has(getLayer(o)))for(const m of Array.isArray(o.material)?o.material:[o.material])shared.add(m);
  const own=(m:THREE.Material)=>{if(!shared.has(m))return m;let c=copies.get(m);if(!c){c=m.clone();atmosphere.setupMaterial(c);copies.set(m,c);}return c;};
  for(const o of objects)if(!ranked.has(getLayer(o))){o.material=Array.isArray(o.material)?o.material.map(own):own(o.material);originals.set(o,o.material);}}
 // Each material class gets the finish of the surface it names; glazing, metal, paint and asphalt reflect a local probe.
 const probeMaterials=finishMaterials(objects);
 // Window and door reveals are drawn by the generator as a near-black 'recess' surface, which outlined every opening in thick
 // black; a reveal is the wall turned inward, so each takes its building's main wall material (the occlusion passes shade it
 // as a recess). Reveals are many thin strips; a recess mesh of large panels (a venue's dark backings) keeps its finish.
 {const wallClass=/^(brick|render|stucco|stone|limestone|cast_stone|concrete|terracotta|timber|panel_dark|venue_concrete)/;const area=new Map<string,Map<THREE.Material,number>>();
  for(const o of objects){const l=getLayer(o);if(l!=='buildings'&&l!=='facades')continue;if(!wallClass.test(classOf(o.name))||Array.isArray(o.material))continue;const id=getId(o);if(!id)continue;
   const g=o.geometry,n=(g.index?g.index.count:g.attributes.position.count)/3;let m=area.get(id);if(!m){m=new Map();area.set(id,m);}m.set(o.material,(m.get(o.material)??0)+n);}
  const thin=(o:THREE.Mesh)=>{const g=o.geometry,p=g.attributes.position,idx=g.index,n=idx?idx.count:p.count,A:number[]=[],a=new THREE.Vector3(),b=new THREE.Vector3(),c=new THREE.Vector3();
   for(let i=0;i<n;i+=3){a.fromBufferAttribute(p,idx?idx.getX(i):i);b.fromBufferAttribute(p,idx?idx.getX(i+1):i+1);c.fromBufferAttribute(p,idx?idx.getX(i+2):i+2);A.push(b.sub(a).cross(c.sub(a)).length()/2);}
   A.sort((x,y)=>x-y);return A.length>0&&A[A.length>>1]<.25&&A[A.length-1]<2;};
  for(const o of objects){if(classOf(o.name)!=='recess'||Array.isArray(o.material)||!thin(o))continue;const m=area.get(getId(o));if(!m)continue;let best:THREE.Material|null=null,bn=-1;for(const [mat,n] of m)if(n>bn){best=mat;bn=n;}if(best){o.material=best;originals.set(o,best);}}}
 // One material per surface class where the buildings' own copies differ only by a tint (material-merge.ts): the tint
 // goes into a vertex colour, the copies' maps are never uploaded, and the class batches across buildings below.
 {const built=objects.filter(o=>['buildings','facades','foundations','roofs'].includes(getLayer(o)));
  // the depth priority's layer rank (below): merged only within one rank
  const rankOf=(o:THREE.Mesh)=>{const l=getLayer(o);return l==='facades'||l==='signs'?3:l==='roofs'?2:l==='buildings'?(o.name.includes('__observed_')?1:0):-1;};
  const {replaced}=mergeTintVariants(built,o=>classOf(o.name),rankOf);
  const kept=new Set<THREE.Texture>();for(const o of objects)for(const m of Array.isArray(o.material)?o.material:[o.material])for(const v of Object.values(m))if(v instanceof THREE.Texture)kept.add(v);
  for(const m of replaced){for(const v of Object.values(m))if(v instanceof THREE.Texture&&!kept.has(v))v.dispose();m.dispose();}
  for(const o of objects)originals.set(o,o.material);}
 // Street grime, rain streaks and tone drift on walls, patched membranes on flat roofs (world space, no per-building data).
 const wxGround=groundTexture(data.grid,data.ground);
 // the road field's uniforms (roadfield.ts): its texture is drawn once the world is batched, before the first frame
 const roadField:Record<string,THREE.IUniform>={uRfTex:{value:null},uRfR:{value:1}},paving:Record<string,THREE.IUniform>={uPfTex:{value:null},uPfR:{value:1}};let weathering:ReturnType<typeof weatherMaterials>|null=null;let glazedWindows:GlazedWindow[]=[];
 {const tex=wxGround;const wx=new Set<THREE.MeshStandardMaterial>();
  for(const o of objects){if(!WEATHERED_LAYERS.has(getLayer(o))||UNWEATHERED.test(classOf(o.name)))continue;for(const m of Array.isArray(o.material)?o.material:[o.material])if(m instanceof THREE.MeshStandardMaterial)wx.add(m);}
  // render, concrete and cladding get their own surface character (weathering.ts surfaceKind)
  const kinds=new Map<THREE.Material,number>();for(const o of objects){const k=surfaceKind(classOf(o.name));if(k)for(const m of Array.isArray(o.material)?o.material:[o.material])kinds.set(m,k);}
  weathering=weatherMaterials(wx,tex,data.grid,kinds);
  // A dim room behind every window instead of an opaque black pane.
  const glass=new Set<THREE.MeshStandardMaterial>(),glassMeshes:THREE.Mesh[]=[],glassCls=new Map<THREE.Material,string>();
  for(const o of objects){const c=classOf(o.name);if(!isGlazing(c))continue;glassMeshes.push(o);for(const m of Array.isArray(o.material)?o.material:[o.material])if(m instanceof THREE.MeshStandardMaterial){glass.add(m);glassCls.set(m,c);}}
  // every window its own coordinates and identity, so each gets its own blinds, curtains or room (glazing.ts)
  glazedWindows=paneAttributes(glassMeshes);glazeMaterials(glass,glassCls);
  // Worn asphalt and grown lawns instead of one repeating texture and one flat green (ground-detail.ts).
  const ofClass=(re:RegExp)=>{const out=new Set<THREE.MeshStandardMaterial>();for(const o of objects){if(!re.test(classOf(o.name)))continue;for(const m of Array.isArray(o.material)?o.material:[o.material])if(m instanceof THREE.MeshStandardMaterial)out.add(m);}return out;};
  detailAsphalt(ofClass(ASPHALT_CLASS),roadField);barkMaterials(ofClass(/^bark$/));detailGrass(ofClass(GRASS_CLASS));
  // the bare terrain between buildings is mostly back gardens, not one plain of paving
  const terrain=new Set<THREE.MeshStandardMaterial>();for(const o of objects)if(getLayer(o)==='terrain'&&o.material instanceof THREE.MeshStandardMaterial)terrain.add(o.material);detailGarden(terrain,roadField);
  // sidewalk concrete (not its joints or tactile strips), on its own copy of a material other concrete shares
  const walks=new Map<THREE.Material,THREE.MeshStandardMaterial>();
  for(const o of objects)if(getLayer(o)==='sidewalks'&&classOf(o.name)==='concrete'&&o.material instanceof THREE.MeshStandardMaterial){let m=walks.get(o.material);if(!m){m=o.material.clone();atmosphere.setupMaterial(m);walks.set(o.material,m);}o.material=m;originals.set(o,m);}
  detailPavement(walks.values(),{...roadField,...paving});}
 // Coincident surfaces (a tier wall on the outline over the outer wall's top band, glazing in its curtain panel's
 // plane, an observed skin on its host wall) z-fight: the winner flips per pixel as the camera moves. Each material
 // gets a fixed depth priority instead: facade detail over roof caps over observed skins over host massing, with a
 // stable tie-break between materials of one layer. Only depth-equal surfaces are affected.
 {const layerRank:Record<string,number>={facades:3,signs:3,roofs:2,buildings:0};const rank=new Map<THREE.Material,number>();
  for(const o of objects){const l=getLayer(o);let r=layerRank[l];if(r===undefined)continue;if(l==='buildings'&&o.name.includes('__observed_'))r=1;
   for(const m of Array.isArray(o.material)?o.material:[o.material])rank.set(m,Math.max(rank.get(m)??-1,r));}
  const byName=[...rank.keys()].sort((a,b)=>a.name.localeCompare(b.name));
  byName.forEach((m,i)=>{const r=rank.get(m)!*3+(i%3);if(r>0){m.polygonOffset=true;m.polygonOffsetFactor=-.35*r;m.polygonOffsetUnits=-2*r;m.needsUpdate=true;}});}
 // Draw-call batching. The world is static, so the meshes of one layer, material and attribute layout inside one 300 m
 // cell are stitched into a single geometry for rendering. Each WebGL draw costs the browser's GPU process about 20 us
 // of CPU to translate (it, not the GPU, was the frame's limit), so the cells are large: a material's whole area is
 // usually one draw, at the price of the GPU transforming some vertices off screen. The source meshes stay in the scene, invisible, for picking, bounds, the
 // car's distance field and the evidence colours (which paint the sources: a batch mixes evidence classes).
 const batches:THREE.Mesh[]=[];const batchRoot=new THREE.Group();batchRoot.name='batches';scene.add(batchRoot);{
  const groups=new Map<string,THREE.Mesh[]>();const box=new THREE.Box3(),centre=new THREE.Vector3();
  for(const o of objects){if(Array.isArray(o.material))continue;const g=o.geometry;box.setFromObject(o);box.getCenter(centre);const cell=Math.floor(centre.x/300)+','+Math.floor(centre.z/300);const key=[getLayer(o),o.material.uuid,g.index?'i':'n',Object.keys(g.attributes).sort().join(','),cell].join('|');const list=groups.get(key);if(list)list.push(o);else groups.set(key,[o]);}
  for(const [key,list] of groups){
   if(list.length<2)continue;
   const parts=list.map(o=>{o.updateWorldMatrix(true,false);const g=o.geometry.clone();g.applyMatrix4(o.matrixWorld);return g;});
   const merged=mergeGeometries(parts,false);parts.forEach(g=>g.dispose());
   if(!merged)continue;
   const first=list[0];const batch=new THREE.Mesh(merged,first.material);batch.name='batch|'+key;batch.castShadow=first.castShadow;batch.receiveShadow=true;
   batch.userData.layer=getLayer(first);batch.userData.evidence=first.userData.evidence;batch.userData.original=first.material;
   batch.userData.members=list;batchRoot.add(batch);batches.push(batch);for(const o of list){o.visible=false;o.userData.batched=batch;}
  }
 }
 // Dense leaf-cluster canopies grown on the pipeline's leaf quads (which are hidden; the layer toggle handles both).
 {const leaves=objects.filter(o=>getLayer(o)==='vegetation'&&classOf(o.name).startsWith('leaf')&&!o.userData.batched);
  if(leaves.length){const foliage=buildFoliage(leaves);atmosphere.setupMaterial(foliage.material);finishFoliage(foliage.material as THREE.MeshStandardMaterial);scene.add(foliage);foliage.userData.original=foliage.material;for(const o of leaves){o.visible=false;o.userData.batched=foliage;}batches.push(foliage);}}
 // The occlusion pass draws depth and normals without alpha: every leaf card would be a solid square casting dark
 // halos through the crowns, so the canopies are left out of that pass (they still receive the result).
 const foliageBatches=batches.filter(b=>b.name==='vegetation___foliage');
 const renderables=[...batches,...objects.filter(o=>!o.userData.batched)];
 // Depth-only passes (the shadow cascades and the occlusion normals) never read a material, so they draw merged
 // copies of the static world (render-budget.ts), a hundred-odd meshes instead of one per material and cell; the
 // scene pass is unchanged. The occlusion pass also stops redrawing the shadow maps it never reads.
 const budget={proxies:true,glass:true,carLod:true};
 const staticMeshes=[...batches.filter(b=>!foliageBatches.includes(b)),...objects.filter(o=>!o.userData.batched&&o.visible)];
 // Thin facade detail, and what stands behind glass or is a few millimetres tall: glass, room interiors, blinds and the
 // tactile studs cast no shadow (glass passes the sun; the rest is inside or too small); frames, mullions, trims and seats
 // stand in for the depth passes only near the camera (render-budget.ts).
 const thin=/^(aluminum|steel|silver|charcoal|white|chrome|seats)$/,inside=/^(glass|glass_dark|glass_light|glass_blue|glass_clear|interior|curtain|screen|yellow)$/;
 const clsOf=(o:THREE.Object3D)=>classOf(((o.userData.members as THREE.Mesh[]|undefined)?.[0]??o).name);   // a batch's: its members'
 for(const o of staticMeshes)if(inside.test(clsOf(o)))o.castShadow=false;
 const isDetail=(o:THREE.Mesh)=>{const c=clsOf(o),l=getLayer(o);return (l==='facades'||l==='roofs'||l==='street_objects')&&(thin.test(c)||inside.test(c));};
 const depth=buildDepthProxies(staticMeshes,getLayer,120,isDetail);scene.add(depth.root);
 const probeSkip=new Set<THREE.Object3D>(staticMeshes.filter(o=>thin.test(clsOf(o))));
 const proxiedStatics=depth.skipped.length?staticMeshes.filter(o=>!depth.skipped.includes(o)):[batchRoot,model.scene];
 const depthPasses=installDepthProxies({renderer,scene,ao,proxies:depth.root,statics:proxiedStatics,aoHidden:()=>horizon?[...foliageBatches,horizon.group]:foliageBatches,enabled:()=>budget.proxies,near:()=>camera.position,detailM:50});
 const shown=(o:THREE.Object3D)=>{const b=o.userData.batched as THREE.Mesh|undefined;return b?b.visible:o.visible;};
 // Street-local surroundings are needed for glossy materials; a sky-only
 // environment makes glazing and metal read as flat painted surfaces.
 // No mipmaps: the filter reads the cube at its own size, and generating them for a half-float cube stalled the GPU for
 // 30-100 ms in the frame each capture completed.
 const reflectionTarget=new THREE.WebGLCubeRenderTarget(128,{type:THREE.HalfFloatType,generateMipmaps:false,minFilter:THREE.LinearFilter});
 // far enough to take in the sky dome (800 m half-size) and the cloud deck: at 420 m the probe held a flat background colour
 // where windows should reflect the sky and its clouds
 const reflectionCamera=new THREE.CubeCamera(.35,2000,reflectionTarget);
 const reflectionFilter=new THREE.PMREMGenerator(renderer);
 let localReflection:THREE.WebGLRenderTarget|null=null,probeSpare:THREE.WebGLRenderTarget|null=null;
 const lastReflectionPosition=new THREE.Vector3(Infinity,Infinity,Infinity);
 let lastReflectionTime=-Infinity;
 const reflectiveMaterials:THREE.MeshStandardMaterial[]=probeMaterials;
 // Parked cars along the kerbs, when the world brings them (parked-cars.ts): four batched draws, their paint and glass lit
 // by the street probe like the world's other glossy surfaces.
 const parked=carsUrl&&/^\/compare\/[\w.-]+\.json$/.test(carsUrl)?await loadParkedCars(carsUrl):null;
 if(parked){scene.add(parked.group);for(const m of parked.materials)atmosphere.setupMaterial(m);reflectiveMaterials.push(...parked.reflective);}
 // Street lamps, trolleybus poles and wires, when the world brings them (overhead.ts).
 const overhead=overheadUrl&&/^\/compare\/[\w.-]+\.json$/.test(overheadUrl)?await loadOverhead(overheadUrl):null;
 if(overhead){scene.add(overhead.group);for(const m of overhead.materials)atmosphere.setupMaterial(m);}
 // The backdrop round the area, when one is built for it (horizon.ts): the real ground and water out to 60 km, the bay, the
 // hills and mountains round it; San Francisco's land beyond the model is plain ground that fades into the haze.
 const horizon=await loadHorizon(data.center,m=>atmosphere.setupMaterial(m),(scene.fog as THREE.FogExp2).color,atmosphere.skyEnvironment).catch(()=>null);if(horizon)scene.add(horizon.group);
 // A capture draws the six cube faces one per frame (all six in one frame was a 15-25 ms hitch every few seconds on the
 // move); with the last face it filters them and hands the result to the glossy surfaces. The filtered probe alternates
 // between two targets, so the one written was last read by the GPU seconds ago: filtering into the one in use made the
 // driver drain the frames still in flight first, a 30-100 ms stall at every capture. A forced capture (load, review
 // captures) draws all six faces at once.
 let captureFace=-1,probePending:THREE.WebGLRenderTarget|null=null;   // -1 idle, else the last face drawn
 function handOver(){
  if(!probePending)return;const first=!localReflection;probeSpare=localReflection;localReflection=probePending;probePending=null;
  for(const material of reflectiveMaterials){material.envMap=localReflection.texture;if(first){material.envMapIntensity=(material.userData.envIntensity as number|undefined)??.7;material.needsUpdate=true;}}
 }
 function filterProbe(){probePending=probeSpare?reflectionFilter.fromCubemap(reflectionTarget.texture,probeSpare):reflectionFilter.fromCubemap(reflectionTarget.texture);}
 function updateReflections(now:number,force=false){
  if(force)captureFace=-1;
  else if(captureFace<0){
   // Street, car and low orbit views (glazing reflects the street it stands in); a high aerial view keeps the sky probe.
   const low=camera.position.y-groundAt(camera.position.x,camera.position.z)<60;
   if((mode!=='street'&&mode!=='car'&&!(mode==='orbit'&&low))||now-lastReflectionTime<4000||camera.position.distanceTo(lastReflectionPosition)<45)return;
  }
  if(captureFace<0){reflectionCamera.position.set(camera.position.x,groundAt(camera.position.x,camera.position.z)+2.5,camera.position.z);lastReflectionPosition.copy(camera.position);lastReflectionTime=now;}
  const previousTarget=renderer.getRenderTarget(),previousFace=renderer.getActiveCubeFace(),previousMip=renderer.getActiveMipmapLevel(),ringVisible=ring.visible,highlightVisible=highlight.visible;
  const shadowAuto=renderer.shadowMap.autoUpdate;
  ring.visible=false;highlight.visible=false;renderer.shadowMap.autoUpdate=false;
  // The probe is a blurred reflection: small objects, thin facade detail (under a texel of it) and the car are left out of
  // its six renders.
  const hidden=renderables.filter(o=>o.visible&&(['vegetation','street_objects','signs','markings'].includes(String(o.userData.layer||getLayer(o)))||probeSkip.has(o)));const parkedWasVisible=parked?.group.visible??false,overheadWasVisible=overhead?.group.visible??false;if(parked)parked.group.visible=false;if(overhead)overhead.group.visible=false;hidden.forEach(o=>o.visible=false);const carWasVisible=car?.group.visible??false;if(car)car.group.visible=false;
  // Avoid feedback: glossy surfaces in the faces reflect the sky environment, not the probe being replaced.
  const current=localReflection?.texture??null;
  for(const material of reflectiveMaterials)material.envMap=null;
  try{
   if(reflectionCamera.coordinateSystem!==renderer.coordinateSystem){reflectionCamera.coordinateSystem=renderer.coordinateSystem;reflectionCamera.updateCoordinateSystem();}
   reflectionCamera.updateMatrixWorld();
   for(const face of force?[0,1,2,3,4,5]:[captureFace+1]){renderer.setRenderTarget(reflectionTarget,face);renderer.render(scene,reflectionCamera.children[face] as THREE.Camera);}
   captureFace=force?5:captureFace+1;
   if(captureFace===5&&!force){filterProbe();handOver();captureFace=-1;}
  }finally{for(const material of reflectiveMaterials)material.envMap=current;ring.visible=ringVisible;highlight.visible=highlightVisible;hidden.forEach(o=>o.visible=true);if(parked)parked.group.visible=parkedWasVisible;if(overhead)overhead.group.visible=overheadWasVisible;if(car)car.group.visible=carWasVisible;renderer.shadowMap.autoUpdate=shadowAuto;renderer.setRenderTarget(previousTarget,previousFace,previousMip);}
  if(force){filterProbe();handOver();captureFace=-1;}
 }
 // The study-radius ring at the world's own radius, draped on the terrain: a flat ring at one height floated over or
 // sank under a hilly area's streets.
 const studyR=data.studyRadiusM??200,segs=Math.max(160,Math.round(studyR*2)),rp:number[]=[],ri:number[]=[];
 for(let k=0;k<=segs;k++){const a=k/segs*Math.PI*2,c=Math.cos(a),s=Math.sin(a);for(const r of [studyR-.35,studyR]){const x=c*r,z=s*r;rp.push(x,groundAt(x,z)+.22,z);}if(k<segs){const i=k*2;ri.push(i,i+1,i+2,i+1,i+3,i+2);}}
 const ringGeometry=new THREE.BufferGeometry();ringGeometry.setAttribute('position',new THREE.Float32BufferAttribute(rp,3));ringGeometry.setIndex(ri);
 const ring=new THREE.Mesh(ringGeometry,new THREE.MeshBasicMaterial({color:'#e3c78b',transparent:true,opacity:.65,side:THREE.DoubleSide}));scene.add(ring);
 const highlight=new THREE.Box3Helper(new THREE.Box3(),new THREE.Color('#f5c76b'));highlight.visible=false;scene.add(highlight);
 function getLayer(o:THREE.Object3D){return String(o.userData.layer||o.name.split('___')[0].split('__')[0])}
 function getId(o:THREE.Object3D){return String(o.userData.featureId||o.name.split('___')[0].split('__')[1]||'')}
 function groundAt(x:number,z:number){const y=-z;const {min,step,n}=data.grid;const px=Math.max(0,Math.min(n-1.00001,(x-min)/step)),py=Math.max(0,Math.min(n-1.00001,(y-min)/step)),ix=Math.floor(px),iy=Math.floor(py),fx=px-ix,fy=py-iy;const a=data.ground[iy*n+ix],b=data.ground[iy*n+ix+1],c=data.ground[(iy+1)*n+ix],d=data.ground[(iy+1)*n+ix+1];return fx+fy<=1?a+(b-a)*fx+(c-a)*fy:d+(c-d)*(1-fx)+(b-d)*(1-fy)}

 function highlightId(id:string){const box=new THREE.Box3();objects.filter(o=>getId(o)===id&&getLayer(o)==='buildings').forEach(o=>box.expandByObject(o));if(!box.isEmpty()){highlight.box.copy(box);highlight.visible=true;}onSelect(id)}
 let driving=false,routePoint=0,driveLookYaw=0;
 let car:Car|null=null,field:ReturnType<typeof buildDistanceField>|null=null;
 const reviewData=await fetch('/review-routes.json').then(r=>r.ok?r.json():{routes:[]}).catch(()=>({routes:[]})) as {routes:number[][][]};
 const reviewRoute=reviewData.routes[0]||[];
 let mode='street',yaw=.785,pitch=0,drag=false,previousX=0,previousY=0,startX=0,startY=0,station=0;
 const keys=new Set<string>();let targetPosition:THREE.Vector3|null=null,targetLook:THREE.Vector3|null=null;
 function view(v:string){driving=false;if(car&&mode==='car')car.stop();mode=v;controls.enabled=v!=='street';keys.clear();
  if(v==='street'){targetPosition=null;targetLook=null;const p=data.streetViews?.[station];const x=p?.position[0]??37,z=-(p?.position[1]??-38);camera.position.set(x,groundAt(x,z)+1.25,z);camera.rotation.order='YXZ';if(p){camera.lookAt(p.lookAt[0],groundAt(p.lookAt[0],-p.lookAt[1])+p.lookAt[2],-p.lookAt[1]);yaw=camera.rotation.y;pitch=camera.rotation.x;}else{yaw=-.785;pitch=.025;camera.rotation.set(pitch,yaw,0);}highlight.visible=false;}
  else if(v==='top'){controls.maxPolarAngle=Math.PI*.49;targetPosition=new THREE.Vector3(0,490,.1);targetLook=new THREE.Vector3(0,0,0)}
  else{controls.maxPolarAngle=Math.PI*.49;targetPosition=new THREE.Vector3(150,115,185);targetLook=new THREE.Vector3(15,5,-8)}
 }
 function select(id:string){const f=data.buildings.find(b=>b.id===id);if(!f)return;highlightId(id);mode='orbit';controls.enabled=true;targetLook=new THREE.Vector3(f.center[0],f.center[2]+f.height*.35,-f.center[1]);targetPosition=targetLook.clone().add(new THREE.Vector3(65,40,75));}
 function drive(on:boolean){
  if(!on){const url=new URL(window.location.href);url.searchParams.set('reviewStep',String(routePoint));window.history.replaceState(null,'',url);}
  driving=on&&reviewRoute.length>1;
  if(driving){driveLookYaw=0;pitch=.015;mode='street';controls.enabled=false;keys.clear();targetPosition=null;targetLook=null;const p=reviewRoute[routePoint];camera.position.set(p[0],groundAt(p[0],-p[1])+1.25,-p[1]);}
 }
 function nextStreet(){station=(station+1)%Math.max(1,data.streetViews?.length||1);view('street');}
 // A drivable car: the player's own vehicle on the mapped roads, with building collision from the foundation
 // outlines. Entering car mode parks the walking and touring cameras; leaving it leaves the car where it stands.
 function routeHeading(p:number[],q:number[]){return Math.atan2(-(q[0]-p[0]),q[1]-p[1]);}
 function nearestRoutePoint(x:number,z:number){let best=0,bestD=Infinity;for(let i=0;i<reviewRoute.length;i++){const d=Math.hypot(reviewRoute[i][0]-x,-reviewRoute[i][1]-z);if(d<bestD){bestD=d;best=i;}}return best;}
 function driveCar(on:boolean){
  if(!on){if(car&&mode==='car'){car.stop();view('street');}return;}
  driving=false;mode='car';controls.enabled=false;keys.clear();targetPosition=null;targetLook=null;highlight.visible=false;
  if(!car)return;car.group.visible=true;if(carPlaced){const [x,z,yaw0]=car.position();car.placeAt(x,z,yaw0);}else{resetCar();carPlaced=true;}
 }
 // The car starts on the review route when that route runs on this world's roads and passes nearby (another area shares
 // the pilot's origin, and a point or two of the pilot's route lands on its streets or in its park), otherwise on the nearest straight stretch of carriageway
 // (car-spawn.ts), clear of parked cars and poles.
 let roads:RoadSpots|null=null,routeHere=false,spawnClear:(x:number,z:number)=>boolean=()=>true;
 function resetCar(){
  if(!car)return;const anchor=car.group.visible&&mode==='car'?car.position():[camera.position.x,camera.position.z,0];
  if(reviewRoute.length>1){const i=nearestRoutePoint(anchor[0],anchor[1]);const p=reviewRoute[i],q=reviewRoute[(i+1)%reviewRoute.length];if(!roads||routeHere&&Math.hypot(p[0]-anchor[0],-p[1]-anchor[1])<60){car.placeAt(p[0],-p[1],routeHeading(p,q));return;}}
  const spot=roads?spawnOnRoad(roads,anchor[0],anchor[1],spawnClear,anchor[2]):null;
  if(spot)car.placeAt(spot[0],spot[1],spot[2]);else car.placeAt(anchor[0],anchor[1],anchor[2]);
 }
 camera.position.set(102,74,125);controls.target.set(1,12,0);controls.update();
 const target=[...data.buildings].sort((a,b)=>Math.hypot(a.center[0],a.center[1])-Math.hypot(b.center[0],b.center[1]))[0];if(target)highlightId(target.id);view('street');
 const savedStep=Number(new URL(window.location.href).searchParams.get('reviewStep'));
 if(savedStep>0&&savedStep<reviewRoute.length){routePoint=Math.floor(savedStep);drive(true);drive(false);const next=reviewRoute[(routePoint+1)%reviewRoute.length];yaw=Math.atan2(-(next[0]-camera.position.x),next[1]+camera.position.z);pitch=.015;camera.rotation.set(pitch,yaw,0);}
 // Reproducible arbitrary inspection cameras use metric ENU coordinates. These
 // are review inputs, never building-specific generation rules.
 const reviewParams=new URL(window.location.href).searchParams;
 function reviewVector(key:string){const text=reviewParams.get(key);if(!text)return null;const a=text.split(',').map(Number);return a.length===3&&a.every(Number.isFinite)?new THREE.Vector3(a[0],a[2],-a[1]):null;}
 const reviewFrom=reviewVector('viewFrom'),reviewAt=reviewVector('viewAt');
 if(reviewFrom&&reviewAt&&reviewFrom.distanceTo(reviewAt)>1){mode='orbit';controls.enabled=true;targetPosition=null;targetLook=null;camera.position.copy(reviewFrom);controls.target.copy(reviewAt);controls.update();highlight.visible=false;}
 const ray=new THREE.Raycaster(),pointer=new THREE.Vector2();
 function down(e:PointerEvent){drag=true;startX=previousX=e.clientX;startY=previousY=e.clientY;targetPosition=null;targetLook=null;}
 function move(e:PointerEvent){if(mode==='car'&&drag&&car)car.look(-(e.clientX-previousX)*.004);if(mode==='street'&&drag){yaw-=(e.clientX-previousX)*.004;if(driving)driveLookYaw-=(e.clientX-previousX)*.004;pitch=Math.max(-1.2,Math.min(1.2,pitch-(e.clientY-previousY)*.004));camera.rotation.set(pitch,yaw,0);}previousX=e.clientX;previousY=e.clientY;}
 function up(e:PointerEvent){const wasDragging=drag;drag=false;if(!wasDragging||Math.hypot(e.clientX-startX,e.clientY-startY)>5)return;const r=renderer.domElement.getBoundingClientRect();pointer.set((e.clientX-r.left)/r.width*2-1,-(e.clientY-r.top)/r.height*2+1);ray.setFromCamera(pointer,camera);const hit=ray.intersectObjects(objects,false).find(h=>shown(h.object)&&getId(h.object));if(hit)highlightId(getId(hit.object));}
 function keydown(e:KeyboardEvent){if(e.target instanceof Element&&e.target.closest('input,textarea,[contenteditable="true"],[role="dialog"]'))return;
  if(mode==='car'&&car&&!e.repeat){if(e.code==='KeyC'){car.toggleCamera();e.preventDefault();return;}if(e.code==='KeyR'){resetCar();e.preventDefault();return;}}
  if(['KeyW','KeyA','KeyS','KeyD','ArrowUp','ArrowDown','ArrowLeft','ArrowRight','ShiftLeft','ShiftRight','Space'].includes(e.code)){keys.add(e.code);if(mode==='street'||mode==='car')e.preventDefault();}}
 function keyup(e:KeyboardEvent){keys.delete(e.code)}
 function blur(){keys.clear();drag=false}
 renderer.domElement.addEventListener('pointerdown',down);window.addEventListener('pointermove',move);window.addEventListener('pointerup',up);window.addEventListener('keydown',keydown);window.addEventListener('keyup',keyup);window.addEventListener('blur',blur);
 function applySize(){const w=host.clientWidth,h=host.clientHeight;if(!w||!h)return;maxScale=scaleCap();if(!qualityLocked)renderScale=Math.min(renderScale,maxScale);camera.aspect=w/h;camera.updateProjectionMatrix();atmosphere.resize();renderer.setPixelRatio(renderScale);renderer.setSize(w,h);composer.setPixelRatio(renderScale);composer.setSize(w,h);ao.setSize(Math.max(1,Math.round(w*renderScale*aoScale)),Math.max(1,Math.round(h*renderScale*aoScale)));}
 const resize=new ResizeObserver(applySize);resize.observe(host);
 // The car is built now and hidden, so its shaders compile with the rest of the scene rather than on the first drive.
 // What the car stands on and runs into: the drivable surface, street objects and cells it cannot enter (car-ground.ts),
 // then the building distance field at the surface raster's resolution with those cells marked in.
 // the car's surface and walls cover the world's context radius (a larger area than the pilot's 265 m)
 const reach=Math.max(275,(data.contextRadiusM??265)+10);
 const carGround=buildCarGround(objects,getLayer,groundAt,reach);
 roads=roadSpots(objects,getLayer);{const r=roads;const on=reviewRoute.filter(p=>onRoad(r,p[0],-p[1])).length;routeHere=reviewRoute.length>1&&on>=.9*reviewRoute.length;}   // the route belongs to this world when nearly all of it runs on its roads
{const near:number[]=[];spawnClear=(x,z)=>{const o=carGround.obstacles;for(const k of o.near(x,z,3,near))if(Math.hypot(o.x[k]-x,o.z[k]-z)<o.r[k]+1.4)return false;return true;};}
 // parked cars (three circles each along their length) and street poles are obstacles too
 if(parked||overhead){const o=carGround.obstacles,flat:number[]=[];for(let i=0;i<o.count;i++)flat.push(o.x[i],o.z[i],o.r[i]);carGround.obstacles=new Obstacles(flat.concat(parked?.circles??[],overhead?.circles??[]));carGround.stats.obstacles=carGround.obstacles.count;}
 const fieldStart=performance.now();field=buildDistanceField(objects,getLayer,reach,.25,carGround.blocked);carGround.stats.fieldMs=Math.round(performance.now()-fieldStart);const carSource=await roadster;car=carSource?createCar(scene,host,camera,groundAt,field,carSource,carGround):null;car?.group.traverse(o=>{if(!(o instanceof THREE.Mesh))return;atmosphere.setupMaterial(o.material);for(const m of Array.isArray(o.material)?o.material:[o.material]){if(m instanceof THREE.MeshStandardMaterial&&m.roughness<.5&&!reflectiveMaterials.includes(m)){m.userData.envIntensity=.7;reflectiveMaterials.push(m);}}});car?.stop();let carPlaced=false;
 // Lighter stand-ins for the car (car-lod.ts): merged, simplified parts in the depth-only passes, simplified parts in the
 // scene pass when the car is far from the camera. Built from whatever model loaded, so a rebuilt car needs nothing else.
 const carLod=car?attachCarLod(car.group,{enabled:()=>budget.carLod}):null;if(carLod)depthPasses.addSwap(carLod.depthSwap);
 const glassCandidates=[...renderables,atmosphere.sky,atmosphere.clouds,ring,highlight];const patchUniforms=cacheArrayUniforms(renderer);
 const glass=cullTransmission({renderer,camera,scenePass,candidates:()=>glassCandidates,glassRoot:()=>car?.group??null,enabled:()=>budget.glass});
 // The world does not move: its matrices are composed once, and the matrix walk of every render call skips it (about
 // 2,000 nodes; the scene root is frozen too, or it would make every descendant recompute).
 scene.matrixAutoUpdate=false;for(const r of [model.scene,batchRoot,depth.root,...foliageBatches]){r.updateMatrixWorld(true);r.traverse(o=>{o.matrixAutoUpdate=false;});r.updateMatrixWorld=()=>{};}
 // Warm-up while the loading notice is up: one full frame (buffers uploaded, shadow maps drawn, post-processing compiled),
 // every material texture uploaded, the first reflection capture, then every shader variant compiled against that probe,
 // car included. Without it the first visible frame froze for close to two seconds (about 900 MB of texture uploads and
 // the probe-lit programs), textures still unseen uploaded on first sight, and the first drive compiled the car's.
 // Sky visibility over the whole area (occlusion.ts): street canyons, building bases and courtyards see less of the sky and
 // are lit less by it; every world material, the canopies, the parked cars, the street lamps and wires and the player's car
 // take it.
 // the road field (roadfield.ts): kerb distance, road width and straightness for every road pixel, for lane-aware wear
 // The road field (roadfield.ts: kerb and frontage distances), the windows' columns for the walls around them (run-off
 // under the sills, cracks: facademap.ts) and the sidewalk slabs from the generator's own joints (paving.ts).
 {const layerOf=(o:THREE.Object3D)=>String(o.userData.layer||getLayer(o));
  const rf=buildRoadField(renderer,scene,renderables.filter(o=>layerOf(o)==='roads'),reach+30,2048,renderables.filter(o=>['roofs','buildings'].includes(layerOf(o))));roadField.uRfTex.value=rf.texture;roadField.uRfR.value=reach+30;
  weathering?.setFacadeMap(buildFacadeMap(glazedWindows,reach+30));
  const pf=buildPavingField(renderer,objects.filter(o=>getLayer(o)==='sidewalks'&&classOf(o.name)==='joint'),reach+30);paving.uPfTex.value=pf.texture;paving.uPfR.value=reach+30;}
 const skyVis=buildSkyVisibility(renderer,scene,[atmosphere.sky,atmosphere.clouds,ring,highlight,...(car?[car.group]:[]),...(horizon?[horizon.group]:[])],reach+30);
 weathering?.setHeight(skyVis.height.texture,reach+30);
 {const mats=new Set<THREE.Material>();for(const o of renderables)for(const m of Array.isArray(o.material)?o.material:[o.material])mats.add(m);
  if(parked)for(const m of parked.materials)mats.add(m);car?.group.traverse(o=>{const m=(o as THREE.Mesh).material;if(m)for(const x of Array.isArray(m)?m:[m])mats.add(x);});
  if(overhead)for(const m of overhead.materials)mats.add(m);
  occludeMaterials(mats,skyVis);}
 onProgress('Preparing shaders and textures…');await new Promise(r=>setTimeout(r,0));
 {applySize();if(car)car.group.visible=false;atmosphere.update(performance.now());patchUniforms();composer.render(0);
  const maps=new Set<THREE.Texture>();scene.traverse(o=>{const m=(o as THREE.Mesh).material;if(!m)return;for(const x of Array.isArray(m)?m:[m])for(const v of Object.values(x))if(v instanceof THREE.Texture&&!(v as THREE.CubeTexture).isCubeTexture&&!v.isRenderTargetTexture)maps.add(v);});
  for(const t of maps)renderer.initTexture(t);
  updateReflections(performance.now(),true);if(car)car.group.visible=true;
  // compiled for the target the scene pass draws into: against the canvas it builds tone-mapped variants nothing uses
  renderer.setRenderTarget(scenePass.target);renderer.compile(scene,camera);renderer.setRenderTarget(null);
  // ... and every material drawn once, culling off: the GPU driver builds each pipeline (shader, vertex layout, target
  // format) at its first draw, so what was not in the first view stalled the frame it first appeared in, 0.1-0.3 s while
  // driving. The car is in view, so its glass's transmission copy is built too.
  {const culled:THREE.Object3D[]=[];scene.traverse(o=>{if(o.frustumCulled){o.frustumCulled=false;culled.push(o);}});if(car)car.group.visible=true;
   try{composer.render(0);}finally{culled.forEach(o=>{o.frustumCulled=true;});}
   // and the probe's second target: allocated and drawn with now, not at the first capture on the road
   filterProbe();handOver();composer.render(0);}}
 if(car)car.group.visible=false;(window as unknown as {__car?:unknown}).__car={car,field,ground:carGround,objects,getLayer};
 let frame=0,previous=performance.now();const forward=new THREE.Vector3(),right=new THREE.Vector3(),worldUp=new THREE.Vector3(0,1,0),delta=new THREE.Vector3();
 let frameMs=16,lastCalls=0;renderer.info.autoReset=false;
 // Adaptive quality, for 60 frames a second. Two signals: frames that missed a 60 Hz deadline (an interval over 20 ms),
 // and the time the frame's own work took. The browser holds the frame's WebGL calls back when the GPU is behind, so
 // that time stays near the frame's pure CPU cost (a few ms) while there is headroom and climbs toward the whole frame
 // when there is none: it shows headroom even when the display's refresh caps the frame rate. Missed deadlines or
 // saturated work step the render scale down a level (12 % fewer pixels each), then switch off the ambient occlusion,
 // then the bloom; sustained headroom brings them back in the opposite order. A step up that soon has to be undone bars
 // further steps up for 20 s. Every step reallocates the render targets, so steps are at least 1.5 s apart.
 const LEVELS=[1,.94,.88,.82,.76,.7];let level=0,workMs=6,lastWork=6,lastChange=performance.now(),upBlockedUntil=0,lastUp=-1e9;const late=new Uint8Array(90);let lateCount=0,lateAt=0,settle=60;   // the first second after load is no measure
 const scaleAt=(l:number)=>Math.max(.5,maxScale*LEVELS[l]);
 function adapt(now:number,raw:number){
  if(qualityLocked||raw>250){settle=10;return;}                       // a hidden or throttled tab is no measure
  if(settle>0){settle--;return;}                                         // the frames after a step reallocate targets
  const miss=raw>20?1:0;lateCount+=miss-late[lateAt];late[lateAt]=miss;lateAt=(lateAt+1)%late.length;
  workMs+=(lastWork-workMs)*.08;
  if(now-lastChange<1500)return;
  const missRate=lateCount/late.length;
  const step=(down:boolean)=>{lastChange=now;settle=10;late.fill(0);lateCount=0;workMs=down?8:6;
   if(down){if(now-lastUp<3000)upBlockedUntil=now+20000;
    if(level<LEVELS.length-1){level++;renderScale=scaleAt(level);applySize();}else if(ao.enabled)ao.enabled=false;else if(grade.bloom.enabled)grade.bloom.enabled=false;}
   else{lastUp=now;if(!grade.bloom.enabled)grade.bloom.enabled=true;else if(!ao.enabled)ao.enabled=true;else if(level>0){level--;renderScale=scaleAt(level);applySize();}}};
  if(missRate>.06||workMs>13.5)step(true);
  else if(now-lastChange>4000&&now>upBlockedUntil&&missRate<.012&&workMs<8&&(level>0||!ao.enabled||!grade.bloom.enabled))step(false);
 }
 function animate(now:number){frame=requestAnimationFrame(animate);const raw=now-previous;const dt=Math.min(raw/1000,.05);previous=now;const workStart=performance.now();
  if(raw<250)frameMs+=(raw-frameMs)*.06;adapt(now,raw);
  if(mode==='street'){
   if(driving){
    let remaining=dt*10;
    while(remaining>0){
     const next=reviewRoute[(routePoint+1)%reviewRoute.length];
     delta.set(next[0]-camera.position.x,0,-next[1]-camera.position.z);const distance=delta.length();
     if(distance<.001){routePoint=(routePoint+1)%reviewRoute.length;remaining=0;break;}
     const amount=Math.min(distance,remaining);camera.position.addScaledVector(delta,amount/distance);remaining-=amount;
     if(!drag){yaw=Math.atan2(-delta.x,-delta.z)+driveLookYaw;camera.rotation.set(pitch,yaw,0);}
     if(amount>=distance-.00001)routePoint=(routePoint+1)%reviewRoute.length;
    }
   }
   camera.getWorldDirection(forward);forward.y=0;forward.normalize();right.crossVectors(forward,worldUp);const speed=dt*(keys.has('ShiftLeft')||keys.has('ShiftRight')?18:5);
   if(keys.has('KeyW')||keys.has('ArrowUp'))camera.position.addScaledVector(forward,speed);if(keys.has('KeyS')||keys.has('ArrowDown'))camera.position.addScaledVector(forward,-speed);if(keys.has('KeyD')||keys.has('ArrowRight'))camera.position.addScaledVector(right,speed);if(keys.has('KeyA')||keys.has('ArrowLeft'))camera.position.addScaledVector(right,-speed);
   const len=Math.hypot(camera.position.x,camera.position.z);if(len>255){camera.position.x*=255/len;camera.position.z*=255/len;}camera.position.y=groundAt(camera.position.x,camera.position.z)+1.25;
  }else if(mode==='car'&&car){
   const input:CarInput={throttle:(keys.has('KeyW')||keys.has('ArrowUp'))?1:0,brake:(keys.has('KeyS')||keys.has('ArrowDown'))?1:0,steer:((keys.has('KeyA')||keys.has('ArrowLeft'))?1:0)-((keys.has('KeyD')||keys.has('ArrowRight'))?1:0),handbrake:keys.has('Space')};
   car.update(dt,input);if(!drag)car.relax(dt);
  }else{if(targetPosition&&targetLook){camera.position.lerp(targetPosition,.07);controls.target.lerp(targetLook,.07);if(camera.position.distanceTo(targetPosition)<.05){targetPosition=null;targetLook=null;}}controls.update();}
  atmosphere.update(now);horizon?.update(now);
  updateReflections(now);
  carLod?.update(camera);glass.prepare();patchUniforms();renderer.info.reset();composer.render(dt);lastCalls=renderer.info.render.calls;
  lastWork=performance.now()-workStart;
 }
 frame=requestAnimationFrame(animate);
 // Review hook: `pose` puts the orbit camera at metric ENU coordinates (as ?viewFrom/?viewAt do) without a reload,
 // `capture` renders one frame and returns it as a PNG blob; both are for same-pose A/B comparisons only. `pose` lifts
 // the orbit's ground limit, which would otherwise raise a street-level camera looking up at a facade to its target's height.
 (window as unknown as {__view?:unknown}).__view={quality:(scale?:number,lock=true)=>{qualityLocked=lock;if(scale){renderScale=scale;applySize();}return {renderScale,maxScale,level,workMs:+workMs.toFixed(1),ao:ao.enabled,bloom:grade.bloom.enabled};},exposure:()=>grade.exposure.read(renderer),pose:(from:number[],at:number[])=>{grade.snap();mode='orbit';controls.enabled=true;controls.maxPolarAngle=Math.PI*.98;targetPosition=null;targetLook=null;highlight.visible=false;camera.position.set(from[0],from[2],-from[1]);controls.target.set(at[0],at[2],-at[1]);controls.update();},
  capture:(w=1800,h=1125)=>new Promise<Blob|null>(resolve=>{qualityLocked=true;ao.enabled=true;grade.bloom.enabled=true;grade.snap();
   renderer.setPixelRatio(1);renderer.setSize(w,h,false);composer.setPixelRatio(1);composer.setSize(w,h);ao.setSize(Math.round(w*aoScale),Math.round(h*aoScale));camera.aspect=w/h;camera.updateProjectionMatrix();atmosphere.resize();
   // the animation loop may be throttled (hidden pane): place the shadow cascades and the reflection probe for this pose
   controls.update();atmosphere.update(performance.now()+1000);atmosphere.csm.lights[1].shadow.needsUpdate=true;updateReflections(performance.now()+1e7,true);
   carLod?.update(camera);glass.prepare();patchUniforms();composer.render(1/60);renderer.domElement.toBlob(b=>{applySize();resolve(b);},'image/png');}),
  lockQuality:(on:boolean)=>{qualityLocked=on;if(on){ao.enabled=true;grade.bloom.enabled=true;}},
  renderOnce:()=>{const t0=performance.now();composer.render(1/60);renderer.getContext().finish();return performance.now()-t0;},budget:{flags:budget,glass:glass.stats,proxies:depth.meshes.length,proxySkipped:depth.skipped.length,car:carLod?.stats??null},info:()=>({calls:lastCalls,triangles:renderer.info.render.triangles,frameMs:+frameMs.toFixed(1),renderScale:+renderScale.toFixed(2),ao:ao.enabled,bloom:grade.bloom.enabled,batches:batches.length,sceneTriangles:objects.reduce((n,o)=>n+(o.geometry.index?o.geometry.index.count:o.geometry.attributes.position.count)/3,0)}),atmosphere,scene,camera,objects,renderer,composer};
 return {downloadUrl,view,select,lighting,nextStreet,drive,car:driveCar,layer:(layer:string,on:boolean)=>{const wanted=(l:string)=>l===layer||(layer==='street_objects'&&l==='signs');if(wanted('street_objects')){if(parked)parked.group.visible=on;if(overhead)overhead.group.visible=on;}for(const o of objects)if(!o.userData.batched&&wanted(getLayer(o)))o.visible=on;for(const b of batches)if(wanted(String(b.userData.layer)))b.visible=on;for(const p of depth.meshes)if(wanted(String(p.userData.layer)))p.visible=on;},evidence:(on:boolean)=>{const tint=(layer:string,evidence:unknown)=>['roofs','terrain'].includes(layer)?'#56b3c4':evidence==='visual-fit'?'#7caf8c':'#d2a25e';const paint=(o:THREE.Mesh,layer:string,evidence:unknown,original:THREE.Material|THREE.Material[])=>{if(on){let m=evidences.get(o);if(!m){m=new THREE.MeshStandardMaterial({color:tint(layer,evidence),roughness:.9});atmosphere.setupMaterial(m);evidences.set(o,m);}o.material=m;}else o.material=original;};for(const o of objects)paint(o,getLayer(o),o.userData.evidence,originals.get(o)!);for(const b of batches){if(b.name==='vegetation___foliage'){paint(b,String(b.userData.layer),b.userData.evidence,b.userData.original as THREE.Material);continue;}const members=b.userData.members as THREE.Mesh[]|undefined;if(!members)continue;if(on){b.userData.wasVisible=b.visible;b.visible=false;for(const o of members)o.visible=b.userData.wasVisible;}else{b.visible=b.userData.wasVisible??true;for(const o of members)o.visible=false;}}},dispose:()=>{cancelAnimationFrame(frame);car?.dispose();parked?.group.children.forEach(m=>(m as THREE.BatchedMesh).dispose());parked?.materials.forEach(m=>m.dispose());overhead?.group.traverse(o=>{if(o instanceof THREE.Mesh)o.geometry.dispose();});overhead?.materials.forEach(m=>m.dispose());resize.disconnect();controls.dispose();renderer.domElement.removeEventListener('pointerdown',down);window.removeEventListener('pointermove',move);window.removeEventListener('pointerup',up);window.removeEventListener('keydown',keydown);window.removeEventListener('keyup',keyup);window.removeEventListener('blur',blur);objects.forEach(o=>o.geometry.dispose());batches.forEach(b=>{b.geometry.dispose();b.removeFromParent();if(b.name==='vegetation___foliage'){const m=b.material as THREE.MeshStandardMaterial;m.map?.dispose();m.dispose();}});wxGround.dispose();new Set([...originals.values()].flat()).forEach(m=>m.dispose());evidences.forEach(m=>m.dispose());localReflection?.dispose();probeSpare?.dispose();reflectionTarget.dispose();reflectionFilter.dispose();depth.dispose();scenePass.dispose();ao.dispose();grade.dispose();composer.dispose();atmosphere.dispose();ring.geometry.dispose();(ring.material as THREE.Material).dispose();highlight.geometry.dispose();(highlight.material as THREE.Material).dispose();URL.revokeObjectURL(downloadUrl);renderer.dispose();renderer.domElement.remove();}};
}
