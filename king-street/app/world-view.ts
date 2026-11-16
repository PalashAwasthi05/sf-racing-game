import {ProgressiveProbe} from './progressive-probe.ts';
import {createSceneImagePool,releaseParserImageSources} from './scene-image-pool';
import {driveDisplayPackage,manageDriveCells, type DriveCell, type DriveStream} from './drive-stream';
import {prepareSceneDrawables,createDrawableTextureUpload, type SceneDrawables} from './scene-drawables';
import {withWarmupPrimitives,warmupCompileObjects,collectWarmupLights,nearbyTextureWarmup,prewarmCell,createCellPreparationTrace,createStageBackpressure} from './startup-warmup';
import {blockPath, loadBlockLayers, type BlockTrial, type BlockVariant} from './block-trial';
import * as THREE from 'three';
import {loadSceneManifest, sceneForDisplay, sceneStream, withGpuStartupPackage, type SceneManifest} from './scene-package';
import {createStreamTransitions,drawTier,advancePresentationTime} from './stream-transition';
import {createTreeDetailLayer} from './tree-detail';
import {loadFacadeMiddleIndex,createFacadeMiddleController} from './facade-middle';
import {finishSurfaces,finishMidGlass} from './surface-finish.ts';
import {parseSceneStream} from './scene-stream';
import {GLTFLoader} from 'three/addons/loaders/GLTFLoader.js';
import {evidenceTint} from './static-detail';
import {OrbitControls} from 'three/addons/controls/OrbitControls.js';
import {EffectComposer} from 'three/addons/postprocessing/EffectComposer.js';
import {GTAOPass} from 'three/addons/postprocessing/GTAOPass.js';
import type {World} from './king-pilot';
import {createCar,buildDistanceField,loadRoadster,type Car,type CarInput} from './car';
import {carContact} from './car-contact';
import {buildCarGround,Obstacles} from './car-ground';
import {collisionOnly} from './collision-only';
import {loadParkedCars} from './parked-cars';
import {createStreetLights} from './street-lights';
import {loadOverhead} from './overhead';
import {createAtmosphere,loadDaySky,PRESET_ORDER,type Preset} from './atmosphere';
import {classOf} from './finish';
import {groundTexture} from './weathering';
import {type GlazingEmitter} from './glazing';
import {buildFacadeMap} from './facademap';
import {disposeOwnedTextures} from './material-resources';
import {supportsStartupS3tc} from './compressed-textures';
import {finishFoliage,foliageWind,loadLeafTexture} from './foliage';
import {TemporalResolve} from './temporal-resolve';
import {conservativeVisibility} from './conservative-visibility';
import {loadDriveResiduals} from './drive-residuals';
import {reflectStreet} from './street-reflections';
import {DepthProbeCapture} from './street-probe';
import {createGrade} from './grade';
import {MultisampledRenderPass,buildDepthProxies,buildDepthProxiesAsync,staticDepthSwap,installDepthProxies,reuseDepthForOcclusion,cullTransmission,cacheArrayUniforms} from './render-budget';
import {attachCarLod} from './car-lod';
import {roadSpots,spawnOnRoad,onRoad,type RoadSpots} from './car-spawn';
import {SunEffectsPass,SUN_LOOKS} from './sun-effects.ts';
import {buildGrassBlades,type GrassBlades} from './grass-blades.ts';
import {GRASS_CLASS,groundSun} from './ground-detail.ts';
import {buildSkyVisibility,buildSurfaceBounce,occludeMaterials,type SurfaceBounce} from './occlusion';
import {buildRoadField} from './roadfield';
import {buildPavingField} from './paving';
import {loadHorizon} from './horizon';
import {NO_SUN_SHADOW,smallShadowDetail} from './shadow-policy';
import {loadWallFaces,applyWallFaces} from './exterior-winding';
import {detailScreenFraction,frontageScreenFraction} from './frontage-projection';

export async function createWorldView(host:HTMLDivElement,data:World,onSelect:(id:string)=>void,onProgress:(message:string)=>void,modelUrl?:string,pinnedManifest?:SceneManifest,trial?:BlockTrial,initialLight:Preset='afternoon',initialDamp=false,initialView?:()=>string|null,streamIndex?:DriveStream){
 const recordStages=new URLSearchParams(location.search).has('measureStartup');
 const stage=(name:string)=>{if(recordStages)performance.mark('world.'+name);};
 stage('entry');
 const scene=new THREE.Scene();
 const path=trial?blockPath(trial.routeEN):null;
 let comparison:Awaited<ReturnType<typeof loadBlockLayers>>|null=null;
 let blockDistance=0,blockPlaying=false,blockLook=0;
 const daySkyLoad=initialLight==='afternoon'?loadDaySky():undefined;
 const leafTexture=loadLeafTexture();
 const roadster=loadRoadster();   // the car model downloads while the world loads
 const wallFaces=streamIndex?.windingCorrections?loadWallFaces(streamIndex.windingCorrections,streamIndex.sourceWorldHash):undefined;
 // The scene stream and texture decoders run while the daytime sky loads.
 const manifestLoad=modelUrl?Promise.resolve(undefined):pinnedManifest?Promise.resolve(pinnedManifest):loadSceneManifest();
 let loadedHash: string | undefined;
 async function loadModel(){
  const preparing=()=>onProgress('Preparing materials and geometry…');
  if(modelUrl){
   const response=await fetch(modelUrl);if(!response.ok)throw new Error('Geometry preview unavailable');
   const bytes=await response.arrayBuffer();preparing();return new GLTFLoader().parseAsync(bytes,'/');
  }
  const portable = (await manifestLoad)!;
  const s3tc=supportsStartupS3tc(renderer);
  const archive = !streamIndex && s3tc ? await withGpuStartupPackage(portable) : portable;
  const display = streamIndex?driveDisplayPackage(streamIndex.base,s3tc):sceneForDisplay(archive,s3tc);
  const parse = (source: SceneManifest) => {
   loadedHash = source.contentHash;
   return parseSceneStream(sceneStream(source,onProgress,undefined,streamIndex?.sourceWorldHash),source.uncompressedBytes,preparing,{imagePool});
  };
  const candidates=[...new Set([display,sceneForDisplay(archive),archive])];
  for(let i=0;i<candidates.length;i++){
   try{return await parse(candidates[i]);}
   catch(error){if(i===candidates.length-1)throw error;}
  }
  throw new Error('Scene unavailable');

 }
 const modelLoad=loadModel();
 void modelLoad.catch(()=>{}); // Startup reports the failure when joining all assets below.
 // far enough for the backdrop (horizon.ts: the East Bay hills, Mount Tamalpais, Mount Diablo); depth precision near the
 // camera is set by the near plane, not the far one
 const camera=new THREE.PerspectiveCamera(58,1,.18,60000);
 // No canvas antialiasing: the canvas only receives the output pass's full-screen quad (the 3D image is antialiased in
 // its own multisampled target below), so canvas samples were memory and a resolve every frame for nothing.
 const renderer=new THREE.WebGLRenderer({antialias:false,powerPreference:'high-performance'});
 // Render scale adapts to the frame time between 0.7 and the device ratio (capped at 1.25); everything sized from it.
 // The most it renders is 1.25 render pixels per CSS pixel and four million pixels in all: a maximised window on a 5K
 // display at 1.25 was 5.3 million, and the frame's cost grows with them (antialiasing, occlusion, post-processing).
 // Bound framebuffer allocation on the streamed drive, including large Retina windows.
 const imagePool=streamIndex?createSceneImagePool({anisotropy:Math.min(8,renderer.capabilities.getMaxAnisotropy())}):undefined;
 const compactDisplay=streamIndex && new URLSearchParams(location.search).get('display')==='720';
 const PIXEL_BUDGET=streamIndex?(compactDisplay?1280*720:1920*1080):4e6;
 const scaleCap=()=>Math.max(streamIndex ? .01 : .5,Math.min(window.devicePixelRatio,1.25,Math.sqrt(PIXEL_BUDGET/Math.max(1,host.clientWidth*host.clientHeight))));
 let maxScale=scaleCap();let renderScale=maxScale,qualityLocked=false;renderer.setPixelRatio(renderScale);renderer.shadowMap.enabled=true;renderer.shadowMap.type=THREE.PCFSoftShadowMap;renderer.toneMapping=THREE.ACESFilmicToneMapping;renderer.toneMappingExposure=.85;
 host.appendChild(renderer.domElement);renderer.domElement.setAttribute('aria-label','3D neighborhood. Use the camera controls to explore.');
 const controls=new OrbitControls(camera,renderer.domElement);controls.enableDamping=true;controls.dampingFactor=.08;controls.minDistance=4;controls.maxDistance=700;controls.maxPolarAngle=Math.PI*.49;
 // Sky, clouds, sun, sky-lit environment, fog, exposure and cascaded shadows live in one module with three presets.
 const atmosphere=createAtmosphere(scene,renderer,camera,initialLight,await daySkyLoad);
 // The lit scene is drawn into a multisampled target: the composer's default target has none, and the canvas's own
 // antialiasing never touches the 3D image once post-processing is on, so every edge and window frame was aliased.
 // The temporal resolve already samples an eight-position subpixel pattern.
 // A single HDR colour/depth pair avoids a multisampled copy and its full-frame resolve.
 const composer=new EffectComposer(renderer,new THREE.WebGLRenderTarget(1,1,{type:THREE.HalfFloatType,depthBuffer:false}));const scenePass=new MultisampledRenderPass(scene,camera,0,true);composer.setPixelRatio(1);composer.addPass(scenePass);
 const cellWarmTarget=new THREE.WebGLRenderTarget(8,8,{type:THREE.HalfFloatType,depthBuffer:true});
 // Sub-metre contact occlusion uses a small, depth-aware normal buffer
 // reconstructed from the beauty pass. The baked street field covers buildings.
 const ao=new GTAOPass(scene,camera,1,1);ao.updateGtaoMaterial({radius:.7,distanceExponent:1.,thickness:.25,samples:24});ao.updatePdMaterial({samples:12,radius:4,depthPhi:.2,normalPhi:8});ao.blendIntensity=.65;
 // Both AO passes write a single visibility factor. Lighting reads red only.
 for(const target of [ao.gtaoRenderTarget,ao.pdRenderTarget]){target.texture.format=THREE.RedFormat;target.depthBuffer=false;}
 ao.pdMaterial.defines.SAMPLE_LUMINANCE='a.r';
 ao.output=-1;ao.needsSwap=false;   // computed only: the composite applies it
 const sizeOcclusion=ao.setSize.bind(ao);
 ao.setSize=(w,h)=>sizeOcclusion(Math.max(1,Math.round(w*.5)),Math.max(1,Math.round(h*.5)));
 composer.addPass(ao);
 // ... applied to the indirect share of each pixel's light only, with the exposure, in one pass over the scene's own
 // resolved image (no copy of it); then the bloom and the grade, tone mapping and output in one more (grade.ts).
 scenePass.direct=true;
 const temporal=new TemporalResolve(scenePass.target,camera);
 scenePass.temporal=temporal;composer.addPass(temporal);
 const disposeOcclusionResolve=reuseDepthForOcclusion(ao,scenePass.target.depthTexture!,temporal.depthProjectionInverse,temporal.depthProjection);
 let sunEffects:SunEffectsPass|null=null;
 const grade=createGrade(composer,scenePass.target.texture,ao,temporal.source,scenePass.target.depthTexture!,exposure=>sunEffects=new SunEffectsPass(camera,scenePass.target.depthTexture!,temporal.depthProjectionInverse,exposure,atmosphere.csm.lights,()=>atmosphere.sunDirection(),atmosphere.nearShadowM));sunEffects!.look=SUN_LOOKS[atmosphere.current()]??SUN_LOOKS.afternoon;

 let resampleBudget=false;
 const surfaceNight={value:0};
 const roadDamp={value:initialDamp?1:0},roadReflection={value:null as THREE.Texture|null};
 function damp(on:boolean){resampleBudget=true;roadDamp.value=on?1:0;temporal.reset();}
 let bounce:SurfaceBounce|null=null;
 let grass:GrassBlades|null=null;
 function updateBounceSun(){groundSun.value.copy(atmosphere.sunDirection());bounce?.setSun(atmosphere.sunDirection(),atmosphere.csm.lights[0].color,atmosphere.csm.lightIntensity);}
 const glazingEmitters:GlazingEmitter[]=[];
 let syncStreamWindows:(()=>void)|undefined;
 function lighting(preset:string){resampleBudget=true;const p=(PRESET_ORDER as string[]).includes(preset)?preset as Preset:'afternoon';if(atmosphere.current()!==p)atmosphere.apply(p);updateBounceSun();grade.preset(p);if(sunEffects)sunEffects.look=SUN_LOOKS[p]??SUN_LOOKS.afternoon;surfaceNight.value=p==='night'?1:0;streetLights.night(p==='night');if(p==='night')syncStreamWindows?.();grade.snap();temporal.reset();reviewLightReset=true;}
 stage('atmosphere');
 const manifest = await manifestLoad;
 const compareAsset = (url?: string): url is string => !!url && /^\/compare\/[\w.-]+\.json$/.test(url);
 const [model, parked, overhead, horizon, reviewData, distant] = await Promise.all([
  modelLoad,
  compareAsset(manifest?.cars) ? loadParkedCars(manifest.cars,'/vehicle/parked/',renderer) : null,
  compareAsset(manifest?.overhead) ? loadOverhead(manifest.overhead,renderer.extensions.has('WEBGL_multi_draw')) : null,
  loadHorizon(data.center,m=>atmosphere.setupMaterial(m),(scene.fog as THREE.FogExp2).color,atmosphere.skyEnvironment).catch(()=>null),
  streamIndex ? {routes:[streamIndex.routeEN]} : trial ? {routes:[trial.routeEN]} : fetch('/review-routes.json').then(r=>r.ok?r.json():{routes:[]}).catch(()=>({routes:[]})) as Promise<{routes:number[][][]}>,
  streamIndex?.distantExteriors ? loadDriveResiduals(streamIndex,undefined,leafTexture) : null,
 ]);
 stage('assets');
 const wxGround=groundTexture(data.grid,data.ground);
 const roadField:Record<string,THREE.IUniform>={uRfTex:{value:null},uRfR:{value:1},uRfExtent:{value:new THREE.Vector4()}},paving:Record<string,THREE.IUniform>={uPfTex:{value:null},uPfR:{value:1},uPfSize:{value:0}};
 const drawableShared={data,renderer,atmosphere,leafTexture,groundAt,wxGround,roadField,paving,roadDamp,roadReflection,surfaceNight};
 let syncCellCoverage:(id:string)=>void=()=>{};
 let sceneryTime=0;
 const transitions=streamIndex?.scope==='neighborhood-stream'&&!distant
   ?createStreamTransitions(streamIndex.cells.map(cell=>cell.id),id=>syncCellCoverage(id),()=>sceneryTime):null;
 function baseTier(mesh:THREE.Mesh){
   const source=(mesh.userData.members as THREE.Mesh[]|undefined)?.[0]??mesh;
   const layer=getLayer(source);
   return layer==='vegetation'?drawTier.baseTree:layer==='roofs'?drawTier.baseRoof:drawTier.baseFacade;
 }
 function tagBaseMesh(mesh:THREE.Mesh){
   const source=(mesh.userData.members as THREE.Mesh[]|undefined)?.[0]??mesh;
   if(source.userData.class==='distant_exterior'&&source.userData.streamCell)
     transitions?.tag(mesh,source.userData.streamCell,baseTier(mesh));
 }
 const drawables=await prepareSceneDrawables(model,drawableShared,{sceneHash:loadedHash,prepareRenderable:transitions?tagBaseMesh:undefined,stage:name=>{if(name==='materials')stage(name);}});
 const {objects,originals,renderables,submissionMeshes,submissions,probeMaterials,batchRoot,batches,foliageBatches,glazedWindows,weathering}=drawables;
 // Startup light-field bakes temporarily replace individual source materials.
 submissions.setEnabled(false);
 glazingEmitters.push(...drawables.glazingEmitters);
 const evidences=new Map<string,THREE.Material>();
 scene.add(drawables.root);
 if(distant){
   scene.add(distant.root);
   for(const material of distant.materials){
     atmosphere.setupMaterial(material);finishSurfaces([material]);
     if(material.name==='foliage')finishFoliage(material);
   }
   for(const child of distant.root.children){
     const mesh=child as THREE.Mesh;
     mesh.castShadow=!!mesh.userData.persistentFoliage || mesh.userData.layer!=='vegetation' && !(mesh.material instanceof THREE.MeshPhysicalMaterial);
   }
 }
 // Depth-only passes (the shadow cascades and the occlusion normals) never read a material, so they draw merged
 // copies of the static world (render-budget.ts), a hundred-odd meshes instead of one per material and cell; the
 // scene pass is unchanged. The occlusion pass also stops redrawing the shadow maps it never reads.
 const budget={proxies:true,glass:true,carLod:true,visibility:true};
 const alphaCanopies=new Set(foliageBatches);
 const staticMeshes=[...batches,...objects.filter(o=>!o.userData.batched&&o.visible)].filter(mesh=>!alphaCanopies.has(mesh));
 const originalMesh=(m:THREE.Mesh)=>(m.userData.members as THREE.Mesh[]|undefined)?.[0]??m;
 const hullMesh=(m:THREE.Mesh)=>originalMesh(m).userData.class==='collision_hull'||originalMesh(m).name.includes('___collision_hull');
 const exteriorMesh=(m:THREE.Mesh)=>originalMesh(m).userData.class==='distant_exterior';
 const hasExteriorProxy=streamIndex?.scope==='neighborhood-stream';
 const temporaryMesh=(m:THREE.Mesh)=>hullMesh(m)||exteriorMesh(m);
 // Glass passes the sun, but the opaque room backing behind it blocks it.
 // Preserve that backing in distant shadow passes too: without it, sunlight
 // travels through the hollow shell and prints opposite window grids on roads.
 const clsOf=(o:THREE.Object3D)=>classOf(((o.userData.members as THREE.Mesh[]|undefined)?.[0]??o).name);   // a batch's: its members'
 for(const o of staticMeshes)if(NO_SUN_SHADOW.test(clsOf(o)))o.castShadow=false;
 const isDetail=(o:THREE.Mesh)=>smallShadowDetail(clsOf(o),getLayer(o));
 const depth=buildDepthProxies(streamIndex?staticMeshes.filter(m=>!temporaryMesh(m)):staticMeshes,getLayer,54,isDetail,true);scene.add(depth.root);
 if(transitions)for(const mesh of [...renderables,...submissionMeshes])if(mesh.castShadow)transitions.shadow(mesh);
 const skippedDepth=new Set(depth.skipped);
 const proxiedStatics=streamIndex?staticMeshes.filter(m=>!skippedDepth.has(m)&&!temporaryMesh(m)):skippedDepth.size?staticMeshes.filter(o=>!skippedDepth.has(o)):[batchRoot,model.scene];
 const depthPasses=installDepthProxies({renderer,scene,proxies:depth.root,statics:proxiedStatics,enabled:()=>budget.proxies,near:()=>camera.position,detailM:50});
 const shown=(o:THREE.Object3D)=>{const b=o.userData.batched as THREE.Mesh|undefined;return b?b.visible:o.visible;};
 // Street-local surroundings are needed for glossy materials; a sky-only
 // environment makes glazing and metal read as flat painted surfaces.
 // No mipmaps: the filter reads the cube at its own size, and generating them for a half-float cube stalled the GPU for
 // 30-100 ms in the frame each capture completed.
 // Distant reflections still include off-camera geometry. A smaller cube leaves
 // headroom for the approaching block and both filtered reflection atlases.
 const reflectionProbe=new DepthProbeCapture(streamIndex?256:512);
 const probeState={origin:{value:new THREE.Vector3()},enabled:{value:0},capture:{value:0}};
 // far enough to take in the sky dome (800 m half-size) and the cloud deck: at 420 m the probe held a flat background colour
 // where windows should reflect the sky and its clouds
 const reflectionCamera=new THREE.CubeCamera(.35,2000,reflectionProbe.cube);
 // Both streamed probes filter 256-pixel faces on this renderer. Their finished
 // atlases are distinct. The blur scratch can be shared between captures.
 const reflectionFilter=streamIndex?atmosphere.environmentFilter:new THREE.PMREMGenerator(renderer);
 const roughnessFilter=new ProgressiveProbe(reflectionFilter);
 let localReflection:THREE.WebGLRenderTarget|null=null,probeSpare:THREE.WebGLRenderTarget|null=null;
 const lastReflectionPosition=new THREE.Vector3(Infinity,Infinity,Infinity);
 let lastReflectionTime=-Infinity;
 const reflectiveMaterials:THREE.MeshStandardMaterial[]=probeMaterials;
 if(distant)for(const material of distant.materials)if(material instanceof THREE.MeshPhysicalMaterial)reflectiveMaterials.push(material);
 // Parked cars along the kerbs, when the world brings them (parked-cars.ts): four batched draws, their paint and glass lit
 // by the street probe like the world's other glossy surfaces.
 if(parked){scene.add(parked.group);for(const m of parked.materials)atmosphere.setupMaterial(m);reflectiveMaterials.push(...parked.reflective);}
 // Street lamps, trolleybus poles and wires, when the world brings them (overhead.ts).
 const streetLights=createStreetLights(objects,overhead,groundAt);scene.add(streetLights.group);
 if(overhead){scene.add(overhead.group);for(const m of overhead.materials)atmosphere.setupMaterial(m);}
 // The backdrop round the area, when one is built for it (horizon.ts): the real ground and water out to 60 km, the bay, the
 // hills and mountains round it; San Francisco's land beyond the model is plain ground that fades into the haze.
 if(horizon)scene.add(horizon.group);
 // A capture draws the six cube faces one per frame (all six in one frame was a 15-25 ms hitch every few seconds on the
 // move); with the last face it filters them and hands the result to the glossy surfaces. The filtered probe alternates
 // between two targets, so the one written was last read by the GPU seconds ago: filtering into the one in use made the
 // driver drain the frames still in flight first, a 30-100 ms stall at every capture. A forced capture (load, review
 // captures) draws all six faces at once.
 let captureFace=-1,probePending:THREE.WebGLRenderTarget|null=null;   // -1 idle, else the last face drawn
 function handOver(){
  if(!probePending)return;const first=!localReflection;probeSpare=localReflection;localReflection=probePending;roadReflection.value=localReflection.texture;probePending=null;
  if(first){
   // Reserve a matching output for the next moving capture. Its contents
   // are never sampled before that capture has overwritten the whole atlas.
   probeSpare=localReflection.clone();
   renderer.initRenderTarget(probeSpare);
  }
  probeState.origin.value.copy(reflectionCamera.position);probeState.enabled.value=1;
  for(const material of reflectiveMaterials){material.envMap=localReflection.texture;if(first){material.envMapIntensity=(material.userData.envIntensity as number|undefined)??.7;material.needsUpdate=true;}}
 }
 function filterProbe(){probePending=probeSpare?reflectionFilter.fromCubemap(reflectionProbe.cube.texture,probeSpare):reflectionFilter.fromCubemap(reflectionProbe.cube.texture);}
 function updateReflections(now:number,force=false){
  const chasing=mode==='car'&&car?.group.visible;
  const origin=chasing&&car?car.group.position:camera.position;
  if(force){roughnessFilter.cancel();captureFace=-1;}
  else if(roughnessFilter.pending){probePending=roughnessFilter.advance(renderer);handOver();return;}
  else if(captureFace<0){
   // Street, car and low orbit views (glazing reflects the street it stands in); a high aerial view keeps the sky probe.
   const low=camera.position.y-groundAt(camera.position.x,camera.position.z)<60;
   if((mode!=='street'&&mode!=='car'&&!(mode==='orbit'&&low))||now-lastReflectionTime<(chasing?500:1000)||origin.distanceTo(lastReflectionPosition)<(chasing?6:12))return;
  }
  if(captureFace<0){reflectionCamera.position.set(origin.x,chasing?origin.y+.95:groundAt(origin.x,origin.z)+2.5,origin.z);lastReflectionPosition.copy(origin);lastReflectionTime=now;}
  const previousTarget=renderer.getRenderTarget(),previousFace=renderer.getActiveCubeFace(),previousMip=renderer.getActiveMipmapLevel(),ringVisible=ring.visible,highlightVisible=highlight.visible;
  const disc=atmosphere.sky.material.uniforms.uDrawDisc;
  const previousDisc=disc.value,previousWet=roadDamp.value;
  roadDamp.value=0;
  disc.value=0; // The directional light already supplies the solar highlight on paint.
  const shadowAuto=renderer.shadowMap.autoUpdate;
  ring.visible=false;highlight.visible=false;renderer.shadowMap.autoUpdate=false;if(grass)grass.mesh.visible=false;
  // Capture the street from every direction, including markings, poles, glass
  // mullions and parked vehicles. Only exclude the player sitting on the lens.
  const carWasVisible=car?.group.visible??false;if(car)car.group.visible=false;
  probeState.capture.value=1;
  // Avoid feedback: glossy surfaces in the faces reflect the sky environment, not the probe being replaced.
  const current=localReflection?.texture??null;
  for(const material of reflectiveMaterials)material.envMap=null;
  try{
   if(reflectionCamera.coordinateSystem!==renderer.coordinateSystem){reflectionCamera.coordinateSystem=renderer.coordinateSystem;reflectionCamera.updateCoordinateSystem();}
   reflectionCamera.updateMatrixWorld();
   for(const face of force?[0,1,2,3,4,5]:[captureFace+1]){reflectionProbe.renderFace(renderer,scene,reflectionCamera.children[face] as THREE.PerspectiveCamera,face);}
   captureFace=force?5:captureFace+1;
   // Publish the completed cube only AFTER restoring ordinary scene state below.
  }finally{roadDamp.value=previousWet;disc.value=previousDisc;for(const material of reflectiveMaterials)material.envMap=current;ring.visible=ringVisible;highlight.visible=highlightVisible;probeState.capture.value=0;if(grass)grass.mesh.visible=true;if(car)car.group.visible=carWasVisible;renderer.shadowMap.autoUpdate=shadowAuto;renderer.setRenderTarget(previousTarget,previousFace,previousMip);}
  if(captureFace===5){
   if(force||!probeSpare){filterProbe();handOver();}
   else roughnessFilter.start(reflectionProbe.cube.texture,probeSpare);
   captureFace=-1;
  }
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
  if(!car)return;car.group.visible=true;if(carPlaced){const [x,z,yaw0]=car.position();car.placeAt(x,z,yaw0);}
  else{
   // Optional drive bookmark in local east/north metres and compass degrees.
   const fields=new URL(window.location.href).searchParams.get('driveFrom')?.split(',');
   const pose=fields?.length===3&&fields.every(t=>t.trim().length)?fields.map(Number):null;
   if(pose && pose.every(Number.isFinite) && roads && onRoad(roads,pose[0],-pose[1]) && spawnClear(pose[0],-pose[1]))
    car.placeAt(pose[0],-pose[1],-THREE.MathUtils.degToRad(pose[2]));
   else resetCar();
   carPlaced=true;
  }
 }
 // The car starts on the review route when that route runs on this world's roads and passes nearby (another area shares
 // the pilot's origin, and a point or two of the pilot's route lands on its streets or in its park), otherwise on the nearest straight stretch of carriageway
 // (car-spawn.ts), clear of parked cars and poles.
 let roads:RoadSpots|null=null,routeHere=false,spawnClear:(x:number,z:number)=>boolean=()=>true;
 function resetCar(){
  if(!car)return;
  if(streamIndex?.scope==='drive-stream'){const [p,q]=streamIndex.routeEN;car.placeAt(p[0],-p[1],routeHeading(p,q));return;}
  if(streamIndex?.scope==='neighborhood-stream'&&roads){
   const [x,z,heading]=carPlaced?car.position():[camera.position.x,camera.position.z,0];
   const spot=spawnOnRoad(roads,x,z,spawnClear,heading);
   if(spot){car.placeAt(...spot);return;}
  }
  if(path){const p=path.sample(0);car.placeAt(p.x,p.z,p.yaw);return;}
  const anchor=car.group.visible&&mode==='car'?car.position():[camera.position.x,camera.position.z,0];
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
 let reviewLightReset=false;
 function linkedView(){
  const url=new URL(window.location.href),fragment=new URLSearchParams(url.hash.slice(1));
  const params=fragment.has('viewFrom')?fragment:url.searchParams;
  function vec(k:string){const t=params.get(k);if(!t)return null;const v=t.split(',').map(Number);return v.length===3&&v.every(Number.isFinite)?new THREE.Vector3(v[0],v[2],-v[1]):null;}
  const from=vec('viewFrom'),at=vec('viewAt'),f=Number(params.get('viewFov'));
  if(!from||!at||from.distanceTo(at)<=1)return;
  if(Number.isFinite(f)&&f>=10&&f<=120){camera.fov=f;camera.updateProjectionMatrix();}
  if(mode==='car')driveCar(false);if(driving)drive(false);
  mode='orbit';controls.enabled=true;controls.maxPolarAngle=Math.PI*.98;targetPosition=null;targetLook=null;camera.position.copy(from);controls.target.copy(at);controls.update();highlight.visible=false;grade.snap();temporal.reset();reviewLightReset=true;
 }
 linkedView();window.addEventListener('hashchange',linkedView);
 const ray=new THREE.Raycaster(),pointer=new THREE.Vector2();
 function down(e:PointerEvent){drag=true;startX=previousX=e.clientX;startY=previousY=e.clientY;targetPosition=null;targetLook=null;}
 function move(e:PointerEvent){if(mode==='block-replay'&&drag){blockLook-=(e.clientX-previousX)*.004;pitch=THREE.MathUtils.clamp(pitch-(e.clientY-previousY)*.004,-1.25,1.25);placeReplayCamera();}if(mode==='car'&&drag&&car)car.look(-(e.clientX-previousX)*.004);if(mode==='street'&&drag){yaw-=(e.clientX-previousX)*.004;if(driving)driveLookYaw-=(e.clientX-previousX)*.004;pitch=Math.max(-1.2,Math.min(1.2,pitch-(e.clientY-previousY)*.004));camera.rotation.set(pitch,yaw,0);}previousX=e.clientX;previousY=e.clientY;}
 function up(e:PointerEvent){const wasDragging=drag;drag=false;if(trial||!wasDragging||Math.hypot(e.clientX-startX,e.clientY-startY)>5)return;const r=renderer.domElement.getBoundingClientRect();pointer.set((e.clientX-r.left)/r.width*2-1,-(e.clientY-r.top)/r.height*2+1);ray.setFromCamera(pointer,camera);const hit=ray.intersectObjects(objects,false).find(h=>shown(h.object)&&getId(h.object));if(hit)highlightId(getId(hit.object));}
 function keydown(e:KeyboardEvent){if(e.target instanceof Element&&e.target.closest('input,textarea,[contenteditable="true"],[role="dialog"]'))return;
  if(trial&&e.code==='KeyR'&&!e.repeat){manualBlock(true);e.preventDefault();return;}
  if(mode==='car'&&car&&!e.repeat){if(e.code==='KeyM'){car.mute();e.preventDefault();return;}if(e.code==='KeyT'){car.aids();e.preventDefault();return;}if(e.code==='KeyC'){grade.resetMotion();temporal.reset();car.toggleCamera();e.preventDefault();return;}if(e.code==='KeyR'){grade.resetMotion();temporal.reset();resetCar();e.preventDefault();return;}}
  if(['KeyW','KeyA','KeyS','KeyD','ArrowUp','ArrowDown','ArrowLeft','ArrowRight','ShiftLeft','ShiftRight','Space'].includes(e.code)){keys.add(e.code);if(mode==='street'||mode==='car')e.preventDefault();}}
 function keyup(e:KeyboardEvent){keys.delete(e.code)}
 function blur(){keys.clear();drag=false}
 renderer.domElement.addEventListener('pointerdown',down);window.addEventListener('pointermove',move);window.addEventListener('pointerup',up);window.addEventListener('keydown',keydown);window.addEventListener('keyup',keyup);window.addEventListener('blur',blur);
 // A canvas size assignment clears its drawing buffer, even at the same size.
 // ResizeObserver runs after animation callbacks. Queue genuine output changes
 // until the next draw so a layout notification cannot present an empty frame.
 let resizePending=true,outputW=0,outputH=0,outputScale=0,sceneW=0,sceneH=0;
 renderer.domElement.style.width='100%';renderer.domElement.style.height='100%';
 function applySize(){
   const w=host.clientWidth,h=host.clientHeight;
   if(!w||!h)return;
   resizePending=false;maxScale=scaleCap();
   if(!qualityLocked)renderScale=Math.min(renderScale,maxScale);
   if(w!==outputW||h!==outputH||maxScale!==outputScale){
     renderer.setDrawingBufferSize(w,h,maxScale);
     camera.aspect=w/h;camera.updateProjectionMatrix();atmosphere.resize();
     outputW=w;outputH=h;outputScale=maxScale;
   }
   // Dynamic resolution only resizes offscreen passes. The last pass always
   // fills the stable display surface, even on a Retina viewport.
   const sw=Math.max(1,Math.floor(w*renderScale)),sh=Math.max(1,Math.floor(h*renderScale));
   if(sw!==sceneW||sh!==sceneH){composer.setSize(sw,sh);sceneW=sw;sceneH=sh;}
 }
 const resize=new ResizeObserver(()=>{resizePending=true;});resize.observe(host);
 // The car is built now and hidden, so its shaders compile with the rest of the scene rather than on the first drive.
 // What the car stands on and runs into: the drivable surface, street objects and cells it cannot enter (car-ground.ts),
 // then the building distance field at the surface raster's resolution with those cells marked in.
 // the car's surface and walls cover the world's context radius (a larger area than the pilot's 265 m)
 const reach=Math.max(275,(data.contextRadiusM??265)+10);
 const carGround=buildCarGround(objects,getLayer,groundAt,reach);
 roads=roadSpots(objects,getLayer);{const r=roads;const on=reviewRoute.filter(p=>onRoad(r,p[0],-p[1])).length;routeHere=reviewRoute.length>1&&on>=.9*reviewRoute.length;}   // the route belongs to this world when nearly all of it runs on its roads
{const near:number[]=[];spawnClear=(x,z)=>{const o=carGround.obstacles;for(const k of o.near(x,z,3,near))if(Math.hypot(o.x[k]-x,o.z[k]-z)<o.r[k]+1.4)return false;return true;};}
 // parked cars (three circles each along their length) and street poles are obstacles too
 if(parked||overhead){const o=carGround.obstacles,flat:number[]=[];for(let i=0;i<o.count;i++)flat.push(o.x[i],o.z[i],o.r[i]);
  const proxyPoles=objects.reduce((n,o)=>n+(collisionOnly(o)?Number(o.userData.sourcePoleCount)||0:0),0);
  const poleCircles=overhead&&proxyPoles>0&&proxyPoles===overhead.poles?[]:overhead?.circles??[];
  carGround.obstacles=new Obstacles(flat.concat(parked?.circles??[],poleCircles));carGround.stats.obstacles=carGround.obstacles.count;}
 const fieldStart=performance.now();field=buildDistanceField(objects,getLayer,reach,.25,carGround.blocked);carGround.stats.fieldMs=Math.round(performance.now()-fieldStart);streetLights.windows(glazingEmitters,field);const carSource=await roadster;car=carSource?createCar(scene,host,camera,groundAt,field,carSource,carGround,Math.max(1,(data.contextRadiusM??studyR)-3)):null;car?.group.traverse(o=>{if(!(o instanceof THREE.Mesh))return;atmosphere.setupMaterial(o.material);for(const m of Array.isArray(o.material)?o.material:[o.material]){if(m instanceof THREE.MeshStandardMaterial&&(m.roughness<.5||m.name==='paint')&&!reflectiveMaterials.includes(m)){m.userData.envIntensity=m.name==='paint'||m.name==='mirror'?1.12:.85;reflectiveMaterials.push(m);}}});car?.stop();let carPlaced=false;
 // Lighter stand-ins for the car (car-lod.ts): merged, simplified parts in the depth-only passes, simplified parts in the
 // scene pass when the car is far from the camera. Built from whatever model loaded, so a rebuilt car needs nothing else.
 const receivers=new Set<THREE.Material>();
 for(const mesh of renderables)if(['roads','markings','sidewalks','paths','cycleways','terrain','junctions'].includes(getLayer(mesh)))for(const m of Array.isArray(mesh.material)?mesh.material:[mesh.material])receivers.add(m);
 const contact=car?carContact(car.group,receivers):null;
 const carLod=car?attachCarLod(car.group,{enabled:()=>budget.carLod}):null;if(carLod)depthPasses.addSwap(carLod.depthSwap);
 // Occlusion evidence is built after this frame's jitter and uses actual opaque
 // triangles. Only the driving beauty camera skips proven-hidden batch draws.
 let visibilityReady=false,visibilityReview=false;
 stage('collision-and-batches');
 const visibility=conservativeVisibility({camera,meshes:renderables,layerOf:getLayer,enabled:()=>visibilityReady&&budget.visibility&&!visibilityReview&&mode!=='orbit'});
 scenePass.beforeBeauty=()=>visibility.prepare();
 const farMeshes=distant?(distant.root.children as THREE.Mesh[]):[];
 const glassCandidates=[...renderables,...submissionMeshes,...farMeshes,atmosphere.sky,atmosphere.clouds,ring,highlight];const patchUniforms=cacheArrayUniforms(renderer);
 const glass=cullTransmission({renderer,camera,scenePass,candidates:()=>glassCandidates,glassRoot:()=>car?.group??null,enabled:()=>budget.glass});
 // The world does not move: its matrices are composed once, and the matrix walk of every render call skips it (about
 // 2,000 nodes; the scene root is frozen too, or it would make every descendant recompute).
 scene.matrixAutoUpdate=false;for(const r of [model.scene,batchRoot,depth.root,...foliageBatches]){r.updateMatrixWorld(true);r.traverse(o=>{o.matrixAutoUpdate=false;});r.updateMatrixWorld=()=>{};}
 // Warm-up while the loading notice is up: one full frame (buffers uploaded, shadow maps drawn, post-processing compiled),
 // nearby/visible textures uploaded, the first reflection capture, then every shader variant compiled against that probe,
 // car included. Without it the first visible frame froze for close to two seconds (about 900 MB of texture uploads and
 // the probe-lit programs), textures still unseen uploaded on first sight, and the first drive compiled the car's.
 // Sky visibility over the whole area (occlusion.ts): street canyons, building bases and courtyards see less of the sky and
 // are lit less by it; every world material, the canopies, the parked cars, the street lamps and wires and the player's car
 // take it.
 // the road field (roadfield.ts): kerb distance, road width and straightness for every road pixel, for lane-aware wear
 // The road field (roadfield.ts: kerb and frontage distances), the windows' columns for the walls around them (run-off
 // under the sills, cracks: facademap.ts) and the sidewalk slabs from the generator's own joints (paving.ts).
 if(distant||hasExteriorProxy)for(const mesh of renderables)if(hullMesh(mesh))mesh.visible=false;
 const lightingMeshes=distant||hasExteriorProxy?[...renderables.filter(m=>!hullMesh(m)),...farMeshes]:renderables;
 let bakedRoad:ReturnType<typeof buildRoadField>|undefined,bakedPaving:ReturnType<typeof buildPavingField>|undefined,facadeMap:ReturnType<typeof buildFacadeMap>|undefined;
 {const layerOf=(o:THREE.Object3D)=>String(o.userData.layer||getLayer(o));
  const rf=bakedRoad=buildRoadField(renderer,scene,renderables.filter(o=>layerOf(o)==='roads'),reach+30,2048,lightingMeshes.filter(o=>['roofs','buildings'].includes(layerOf(o))));roadField.uRfTex.value=rf.texture;roadField.uRfR.value=reach+30;roadField.uRfExtent.value.copy(rf.uniforms.uRfExtent.value);
  if(weathering){facadeMap=buildFacadeMap(glazedWindows,reach+30);weathering.setFacadeMap(facadeMap);}
  const pf=bakedPaving=buildPavingField(renderer,objects.filter(o=>getLayer(o)==='sidewalks'&&classOf(o.name)==='joint'),reach+30);paving.uPfTex.value=pf.texture;paving.uPfR.value=reach+30;paving.uPfSize.value=pf.uniforms.uPfSize.value;}
 const bakeHidden=[atmosphere.sky,atmosphere.clouds,ring,highlight,streetLights.group,...(parked?[parked.group]:[]),...(overhead?[overhead.group]:[]),...(comparison?[comparison.root]:[]),...(car?[car.group]:[]),...(horizon?[horizon.group]:[])];
 const skyVis=buildSkyVisibility(renderer,scene,bakeHidden,reach+30);
 const diffuseLayers=new Set(['roads','terrain','sidewalks','buildings','roofs','paths','cycleways','junctions']);
 const diffuseSurfaces=lightingMeshes.filter(mesh=>diffuseLayers.has(String(mesh.userData.layer||getLayer(mesh))) &&
  (Array.isArray(mesh.material)?mesh.material:[mesh.material]).some(m=>m instanceof THREE.MeshStandardMaterial && !m.userData.glazed));
 bounce=buildSurfaceBounce(renderer,scene,diffuseSurfaces,bakeHidden,reach+30,skyVis.height.texture);updateBounceSun();
 weathering?.setHeight(skyVis.height.texture,reach+30);
 // real blades on the lawns near the camera (grass-blades.ts)
 {const layerOf=(o:THREE.Object3D)=>String(o.userData.layer||getLayer(o));
  const lawns=renderables.filter(o=>GRASS_CLASS.test(clsOf(o)));
  if(lawns.length){const covering=new Set(['roads','sidewalks','paths','cycleways','junctions','buildings','roofs','markings']);
   const lawnMaterial=[lawns[0].material].flat()[0] as THREE.MeshStandardMaterial;
   grass=buildGrassBlades(renderer,scene,lawns,renderables.filter(o=>!lawns.includes(o)&&covering.has(layerOf(o))),reach+30,lawnMaterial.color,m=>{atmosphere.setupMaterial(m);occludeMaterials([m],skyVis,bounce);});}}
 {const mats=new Set<THREE.Material>();for(const o of lightingMeshes)for(const m of Array.isArray(o.material)?o.material:[o.material])mats.add(m);
  if(parked)for(const m of parked.materials)mats.add(m);car?.group.traverse(o=>{const m=(o as THREE.Mesh).material;if(m)for(const x of Array.isArray(m)?m:[m])mats.add(x);});
  if(overhead)for(const m of overhead.materials)mats.add(m);
  finishSurfaces(mats);occludeMaterials(mats,skyVis,bounce);streetLights.apply(mats);
  const gloss=new Set(reflectiveMaterials);
  for(const m of mats)if(m instanceof THREE.MeshStandardMaterial&&m.userData.groundDetail?.startsWith('asphalt'))gloss.add(m);
  reflectStreet(gloss,temporal,probeState);}
 submissions.setEnabled(true);submissions.releaseBakeBuffers(scene);
 stage('light-fields');
 if(initialLight!=='afternoon')lighting(initialLight);
 function placeReplayCamera(){
  if(!path)return;
  const p=path.sample(blockDistance), ahead=path.sample(Math.min(path.length,blockDistance+4));
  const sampleBehind=path.sample(Math.max(0,blockDistance-4));
  const run=Math.hypot(ahead.x-sampleBehind.x,ahead.z-sampleBehind.z);
  const slope=run>0?Math.atan2(groundAt(ahead.x,ahead.z)-groundAt(sampleBehind.x,sampleBehind.z),run):0;
  camera.position.set(p.x,groundAt(p.x,p.z)+1.25,p.z);camera.rotation.set(pitch+slope,p.yaw+blockLook,0,'YXZ');
 }
 function seekBlock(fraction:number){
  if(!path)return;
  if(car){car.stop();car.group.visible=false;}
  driving=false;blockPlaying=false;mode='block-replay';controls.enabled=false;keys.clear();targetPosition=null;targetLook=null;
  blockDistance=THREE.MathUtils.clamp(fraction,0,1)*path.length;placeReplayCamera();highlight.visible=false;
 }
 function manualBlock(restart:boolean){
  if(!path||!car)return;
  if(mode==='car'&&!restart)return;
  if(restart){blockDistance=0;blockLook=0;pitch=0;}
  blockPlaying=false;driveCar(true);const p=path.sample(blockDistance);car.placeAt(p.x,p.z,p.yaw);
 }
 const block=trial&&path?{
  variant:(variant:BlockVariant)=>comparison?.set(variant),
  manual:manualBlock,
  seek:seekBlock,
  play:()=>{if(mode==='car'&&car){const [x,z]=car.position();blockDistance=path.project(x,z);}seekBlock(blockDistance>=path.length-.05?0:blockDistance/path.length);blockPlaying=true;},
  pause:()=>{blockPlaying=false;},
  look:(angle:number)=>{if(mode==='car'&&car){const [x,z]=car.position();seekBlock(path.project(x,z)/path.length);}blockLook=angle;pitch=.015;placeReplayCamera();},
  state:()=>({mode:mode==='car'?'manual':blockPlaying?'playing':'paused',distance:mode==='car'&&car?path.project(...car.position().slice(0,2) as [number,number]):blockDistance,length:path.length}),
 }:null;
 if(trial){
  if(!car)throw new Error('The comparison car could not be loaded.');
  comparison=loadBlockLayers(trial,renderer);scene.add(comparison.root);camera.layers.enable(1);
  car.toggleCamera();manualBlock(true);
 }
 if(!trial){
  const query=new URL(window.location.href).searchParams;
  const requested=initialView?.()??(query.get('drive')==='1'&&!query.has('viewFrom')&&!window.location.hash?'car':null);
  if(requested==='car')driveCar(true);
  else if(requested){view(requested);if(targetPosition&&targetLook){camera.position.copy(targetPosition);controls.target.copy(targetLook);targetPosition=null;targetLook=null;controls.update();}}
 }
 // The stable road and collision surface stays with this renderer while the car crosses cells.
 // Detailed building meshes replace only the distant hulls owned by that cell.
 const deferredTextures=new Map<THREE.Texture,THREE.Sphere[]>();
 let pendingTextureWarmup:THREE.Texture[]=[],textureWarmAt=0;const textureWarmOrigin=camera.position.clone();
 const drivePath=streamIndex?blockPath(streamIndex.routeEN):null;
 const sightFrustum=new THREE.Frustum(),sightClip=new THREE.Matrix4(),sightDirection=new THREE.Vector3();
 const sightBounds=new Map<DriveCell,THREE.Box3>();
 type FrontageOwner={id:string;box:THREE.Box3;diagonal:number;detailRange:number;distance:number;priority:number;revision:number};
 const frontageBounds=new Map<DriveCell,{owners:FrontageOwner[];bounds:THREE.Box3;maxDiagonal:number;near:number;predicted:number}>();
 const frontageRadius=100;
 if(streamIndex){
   let lowest=Infinity,highest=-Infinity;
   for(const h of data.ground){lowest=Math.min(lowest,h);highest=Math.max(highest,h);}
   for(const b of data.buildings)highest=Math.max(highest,b.center[2]+b.height);
   for(const cell of streamIndex.pieces??streamIndex.cells){
     const [west,south,east,north]=cell.boundsEN;
     sightBounds.set(cell,new THREE.Box3(new THREE.Vector3(west,lowest,-north),new THREE.Vector3(east,highest,-south)));
   }
   if(streamIndex.scope==='neighborhood-stream'){
     const buildingById=new Map(data.buildings.map(b=>[b.id,b]));
     const footprints=new Map<string,THREE.Box3>();
     for(const object of objects){
       if(getLayer(object)!=='foundations'&&object.userData.class!=='distant_exterior')continue;
       const id=getId(object),building=buildingById.get(id);if(!building)continue;
       if(!object.geometry.boundingBox)object.geometry.computeBoundingBox();
       if(!object.geometry.boundingBox)continue;
       object.updateWorldMatrix(true,false);
       const box=object.geometry.boundingBox.clone().applyMatrix4(object.matrixWorld);
       box.max.y=Math.max(box.max.y,building.center[2]+building.height);
       const previous=footprints.get(id);if(previous)previous.union(box);else footprints.set(id,box);
     }
     for(const cell of streamIndex.cells){
       const owners:FrontageOwner[]=[];
       const bounds=new THREE.Box3();let maxDiagonal=0;
       for(const id of cell.stats?.buildingIds??[]){
         const box=footprints.get(id);if(!box)continue;
         const width=Math.max(box.max.x-box.min.x,box.max.z-box.min.z,box.max.y-box.min.y);
         const diagonal=box.min.distanceTo(box.max);
         maxDiagonal=Math.max(maxDiagonal,diagonal);
         owners.push({id,box,diagonal,detailRange:Math.min(65,Math.max(36,width)),distance:Infinity,priority:Infinity,revision:-1});
         bounds.union(box);
       }
       if(owners.length)frontageBounds.set(cell,{owners,bounds,maxDiagonal,near:Infinity,predicted:Infinity});
     }
   }
 }
 const frontageCentre=new THREE.Vector3(),frontageForecast=new THREE.Vector3();
 const previewEye=new THREE.Vector3(),previewRotation=new THREE.Quaternion(),turnRotation=new THREE.Quaternion();
 const unitScale=new THREE.Vector3(1,1,1),turnAxis=new THREE.Vector3(0,1,0);
 const previewClip=new THREE.Matrix4(),previewView=new THREE.Matrix4(),previewFrustum=new THREE.Frustum();
 let predictFrontages=false;
 // Conservative coarse rejection before projecting the eight corners. Large
 // buildings and narrow fields of view remain eligible farther down the road.
 let frontageProjectionRange=0;
 const previousSightClip=new THREE.Matrix4();
 const supportsDetailCompression=supportsStartupS3tc(renderer);
 const driveSight={cameraStation:0,lookStation:0,revision:0,visible:(cell:DriveCell)=>sightFrustum.intersectsBox(sightBounds.get(cell)!),
   nearFrontage:(cell:DriveCell)=>frontageBounds.get(cell)?.near??Infinity,
   hasWindowGeometry:(cell:DriveCell)=>!!middle?.groupFor(cell.id)&&!!transitions&&transitions.value(cell.id,'facade')>.98,
   residentCost:(cell:DriveCell)=>{
     const cost=cell.stats?.residency;
     if(!cost)return cell.uncompressedBytes??0;
     // CPU-side geometry remains available for picking and context recovery.
     const compressed=driveDisplayPackage(cell,supportsDetailCompression)===cell.gpu;
     return cost.geometryBytes*2+(compressed?cost.gpuMipBytes:cost.portableMipBytes);
   },
   predictedFrontage:(cell:DriveCell)=>frontageBounds.get(cell)?.predicted??Infinity,
   frontageScore:streamIndex?.scope==='neighborhood-stream'?(cell:DriveCell)=>{
     const owner=frontageBounds.get(cell);
     if(!owner)return Infinity;
     owner.near=Infinity;owner.predicted=Infinity;
     if(owner.bounds.distanceToPoint(camera.position)>Math.max(frontageRadius,owner.maxDiagonal*frontageProjectionRange))return Infinity;
     let best=Infinity;
     for(const feature of owner.owners){
       const box=feature.box,near=box.distanceToPoint(camera.position);
       if(near>Math.max(frontageRadius,feature.diagonal*frontageProjectionRange))continue;
       // A quick look to the side can expose the adjoining frontage in a
       // single frame. Warm immediate neighbors even at the edge of the view.
       if(near<=12)owner.near=Math.min(owner.near,.9+near/120);
       if(predictFrontages&&previewFrustum.intersectsBox(box)){
         const projected=frontageScreenFraction(box,previewClip,camera.near);
         owner.predicted=Math.min(owner.predicted,box.distanceToPoint(previewEye)/feature.detailRange,
           projected>0?Math.sqrt(detailScreenFraction/projected):Infinity);
       }
       if(!sightFrustum.intersectsBox(box))continue;
       feature.distance=near;feature.revision=driveSight.revision;
       const approaching=box.distanceToPoint(frontageForecast);
       feature.priority=near/feature.detailRange;
       if(feature.priority>.5){
         const fraction=frontageScreenFraction(box,sightClip,camera.near);
         if(fraction>0)feature.priority=Math.min(feature.priority,Math.sqrt(detailScreenFraction/fraction));
       }
       owner.near=Math.min(owner.near,feature.priority,approaching/feature.detailRange);
       box.getCenter(frontageCentre).sub(camera.position);frontageCentre.y=0;
       const alignment=frontageCentre.dot(sightDirection)/Math.max(1e-6,frontageCentre.length());
       best=Math.min(best,near+20*(1-Math.max(-1,Math.min(1,alignment))));
     }
     return best;
   }:undefined};
 function updateDriveSight(){
   camera.getWorldDirection(sightDirection);
   const fx=camera.projectionMatrix.elements[0],fy=camera.projectionMatrix.elements[5];
   // A visible point is at most hypot(1,1/fx,1/fy) times its eye-depth
   // from the eye. Include perspective expansion toward screen edges.
   frontageProjectionRange=1+Math.hypot(1,1/fx,1/fy)*Math.sqrt((1+fx)*(1+fy)/(4*detailScreenFraction));
   frontageForecast.copy(camera.position);
   predictFrontages=false;
   if(mode==='car'&&car){
     const yaw=car.position()[2],lead=THREE.MathUtils.clamp(car.forwardSpeedMps*1.6,-32,32);
     frontageForecast.x-=Math.sin(yaw)*lead;
     frontageForecast.z-=Math.cos(yaw)*lead;
     // Roughly one second of travel covers the measured preparation time at
     // bends. Keep the actual camera pitch and the driver's chosen look angle.
     if(Math.abs(car.forwardSpeedMps)>1){
       predictFrontages=true;
       const turn=THREE.MathUtils.clamp(car.yawRateRadPerSec,-.7,.7);
       const travel=THREE.MathUtils.clamp(car.forwardSpeedMps,-32,32);
       previewEye.copy(camera.position);
       previewEye.x-=Math.sin(yaw+turn*.5)*travel;previewEye.z-=Math.cos(yaw+turn*.5)*travel;
       camera.getWorldQuaternion(previewRotation).premultiply(turnRotation.setFromAxisAngle(turnAxis,turn));
       previewView.compose(previewEye,previewRotation,unitScale).invert();
       previewClip.multiplyMatrices(camera.projectionMatrix,previewView);
       previewFrustum.setFromProjectionMatrix(previewClip);
     }
   }
   sightClip.multiplyMatrices(camera.projectionMatrix,camera.matrixWorldInverse);
   if(!sightClip.equals(previousSightClip)){previousSightClip.copy(sightClip);driveSight.revision++;}
   sightFrustum.setFromProjectionMatrix(sightClip);
   if(drivePath&&streamIndex?.scope==='drive-stream'){
     driveSight.cameraStation=drivePath.project(camera.position.x,camera.position.z);
     driveSight.lookStation=drivePath.project(camera.position.x+40*sightDirection.x,camera.position.z+40*sightDirection.z);
   }
 }
 const hulls=new Map<string,THREE.Mesh[]>();
 for(const mesh of renderables){
   const source=(mesh.userData.members as THREE.Mesh[]|undefined)?.[0]??mesh;
   const owner=source.userData.streamCell;
   if(owner && (hasExteriorProxy?exteriorMesh(mesh):hullMesh(mesh))){
     const group=hulls.get(owner)??[];group.push(mesh);hulls.set(owner,group);
     if(distant)mesh.visible=false;
   }
 }
 const liveCells=new Map<string,SceneDrawables>();
 const detailWalls=new Map<string,Map<string,Set<THREE.Mesh>>>();
 const cellShadows=new Map<string,THREE.Mesh[]>();
 syncStreamWindows=()=>{if(atmosphere.current()==='night')streetLights.windows([...glazingEmitters,...Array.from(liveCells.values()).flatMap(d=>d.glazingEmitters)],field);};
 const layerState=new Map<string,boolean>();
 let streamEvidence=false, openingCells=true,firstFramePending=true;
 // Query-only diagnostics; ordinary driving allocates no phase entries.
 const cellTrace=new URL(window.location.href).searchParams.get('profileStream')==='1'?createCellPreparationTrace(128):null;
 // Share a frame allowance across all scenery layers. Independent loaders that
 // each wait for the same frame would otherwise submit several uploads at once.
 const stagingQueue:(()=>void)[]=[];
 const stagePressure=createStageBackpressure(renderer.getContext());
 function grantStage(){
   if(stagingQueue.length&&!stagePressure.allow()){requestAnimationFrame(grantStage);return;}
   const resume=stagingQueue.shift();
   if(stagingQueue.length)requestAnimationFrame(grantStage);
   resume?.();
 }
 const yieldCell=()=>new Promise<void>(resolve=>{stagingQueue.push(resolve);if(stagingQueue.length===1)requestAnimationFrame(grantStage);});
 let treeLayer:Awaited<ReturnType<typeof createTreeDetailLayer>>|null=null;
 let middle:ReturnType<typeof createFacadeMiddleController>|null=null;
 const independentTrees=new Set<string>(),fullIds=new Set<string>(),handoffs=new Set<string>(),middleGroups=new Map<string,THREE.Group>();
 const setupMid=new WeakSet<THREE.Material>();
 const sceneryErrors:string[]=[];
 const stagingLights=collectWarmupLights(scene);
 const stageGeometry=async(meshes:THREE.Mesh[],signal:AbortSignal)=>{
   await yieldCell();signal.throwIfAborted();
   const programs=[...meshes,...(transitions?.shadowWarmup(meshes)??[])];
   await prewarmCell(renderer,scene,camera,scenePass.target,cellWarmTarget,meshes,programs,()=>signal.throwIfAborted(),yieldCell,undefined,{lights:stagingLights});
 };
 syncCellCoverage=id=>{
   if(!transitions)return;
   for(const mesh of hulls.get(id)??[])
     mesh.visible=layerState.get(getLayer(mesh))!==false && transitions.shows(id,baseTier(mesh));
   const group=middleGroups.get(id);
   if(group)group.visible=layerState.get('facades')!==false&&transitions.shows(id,drawTier.facade);
 };
 if(transitions && streamIndex){
   await Promise.all([
     createTreeDetailLayer(streamIndex,leafTexture,m=>{atmosphere.setupMaterial(m);finishSurfaces([m]);},{
       prepare:async(cell,signal)=>{
         if(!treeLayer?.readyCells.has(cell.id))await transitions.target(cell.id,'treeFine',cell.quality==='fine',true);
         const tier=cell.quality==='fine'?drawTier.treeFine:drawTier.treeCoarse;
         for(const mesh of cell.renderables){transitions.tag(mesh,cell.id,tier);if(mesh.castShadow)transitions.shadow(mesh);}
         await stageGeometry(cell.renderables,signal);
       },
       replace:(incoming,_previous,signal)=>{
         signal.throwIfAborted();
         return transitions.target(incoming.id,'treeFine',incoming.quality==='fine');
       },
       retire:(cell,signal)=>{
         signal.throwIfAborted();
         return transitions.target(cell.id,'tree',false);
       },
       restore:async(cell,signal)=>{
         signal.throwIfAborted();
         await Promise.all([transitions.target(cell.id,'tree',true),transitions.target(cell.id,'treeFine',cell.quality==='fine')]);
       },
       onChange:()=>{
         if(!treeLayer)return;
         for(const id of treeLayer.readyCells.keys())if(!independentTrees.has(id)){
           independentTrees.add(id);void transitions.target(id,'tree',true,firstFramePending);
         }
         for(const id of independentTrees)if(!treeLayer.readyCells.has(id))independentTrees.delete(id);
       }
     }).then(layer=>{
       treeLayer=layer;scene.add(layer.root);
       for(const id of layer.availableCells)transitions.independentTree(id);
     }).catch(e=>sceneryErrors.push(String(e))),
     streamIndex.facadeMiddle&&loadFacadeMiddleIndex(streamIndex.facadeMiddle,streamIndex).then(index=>{
       middle=createFacadeMiddleController(index,{scene,yieldTask:yieldCell,
         prepare:async(mesh,signal)=>{
           for(const material of Array.isArray(mesh.material)?mesh.material:[mesh.material])if(!setupMid.has(material)){
             setupMid.add(material);atmosphere.setupMaterial(material);finishSurfaces([material]);finishMidGlass(material);occludeMaterials([material],skyVis,bounce);streetLights.apply([material]);
           }
           transitions.tag(mesh,mesh.userData.streamCell,drawTier.facade);if(mesh.castShadow)transitions.shadow(mesh);await stageGeometry([mesh],signal);
         },
         onAttach:(cell,group)=>{
           middleGroups.set(cell.id,group);
           void transitions.target(cell.id,'facade',true,firstFramePending||transitions.value(cell.id,'detail')===1).then(()=>driveSight.revision++);
         },
         onRestore:(cell,group)=>{
         middleGroups.set(cell.id,group);
         void transitions.target(cell.id,'facade',true).then(()=>driveSight.revision++);
       },
       onDetach:async(cell,group,reason,signal)=>{
           await transitions.target(cell.id,'facade',false,reason==='dispose');
           if(!signal.aborted&&middleGroups.get(cell.id)===group){middleGroups.delete(cell.id);driveSight.revision++;}
         }
       });
     }).catch(e=>sceneryErrors.push(String(e))),
   ]);
 }
 const middleEye=new THREE.Vector3(Infinity,Infinity,Infinity),middleLook=new THREE.Vector3();
 const frontageById=new Map([...frontageBounds].map(([cell,frontages])=>[cell.id,frontages]));
 const projectedDemand=(cell:{id:string})=>{
   const owner=frontageById.get(cell.id);let largest=0;
   if(owner)for(const feature of owner.owners){
     if(sightFrustum.intersectsBox(feature.box))largest=Math.max(largest,frontageScreenFraction(feature.box,sightClip,camera.near));
     if(predictFrontages && previewFrustum.intersectsBox(feature.box))largest=Math.max(largest,.7*frontageScreenFraction(feature.box,previewClip,camera.near));
   }
   return Math.sqrt(largest);
 };
 const middleBox=new THREE.Box3(),heightRange=[Infinity,-Infinity];
 for(const box of sightBounds.values()){heightRange[0]=Math.min(heightRange[0],box.min.y);heightRange[1]=Math.max(heightRange[1],box.max.y);}
 let lastMiddleTime=-Infinity,middleFullRevision=-1,fullRevision=0;
 function updateScenery(now:number){
   treeLayer?.update(camera.position,sightDirection);
   if(middle && (middleFullRevision!==fullRevision || now-lastMiddleTime>200 || middleEye.distanceToSquared(camera.position)>4 || middleLook.distanceToSquared(sightDirection)>.004)){
     lastMiddleTime=now;middleFullRevision=fullRevision;middleEye.copy(camera.position);middleLook.copy(sightDirection);
     middle.update({east:camera.position.x,north:-camera.position.z,forwardE:sightDirection.x,forwardN:-sightDirection.z,detailed:fullIds,handoffs,screenDemand:projectedDemand,
       visible:b=>{
         middleBox.min.set(b[0],heightRange[0],-b[3]);middleBox.max.set(b[2],heightRange[1],-b[1]);
         // Keep the immediate corner ready even while its frontage is just outside the windshield.
         return middleBox.distanceToPoint(camera.position)<40 || sightFrustum.intersectsBox(middleBox) ||
           predictFrontages && previewFrustum.intersectsBox(middleBox);
       }});
   }
 }

 const stream=streamIndex?manageDriveCells(streamIndex.pieces??streamIndex.cells,async (cell:DriveCell,signal:AbortSignal)=>{
   const record=(await wallFaces)?.get(cell.id);
   const correction=record?.contentHash===cell.contentHash?record:undefined;
   const preferGpu=supportsStartupS3tc(renderer)&&(!correction||correction.gpuContentHash===cell.gpu?.contentHash);
   const asset=driveDisplayPackage(cell,preferGpu);
   const parseDone=cellTrace?.begin(cell.id,'parse');
   const separateTreeClasses=treeLayer?.coveredClasses.get(cell.id);
   const loaded=await parseSceneStream(sceneStream(asset,()=>{},signal,streamIndex.sourceWorldHash),asset.uncompressedBytes,undefined,{imagePool,omitStreamedTrees:separateTreeClasses,yieldCpu:openingCells?undefined:yieldCell});
   if(correction)loaded.scene.userData.correctedExteriorFaces=applyWallFaces(loaded,correction);
   parseDone?.();let drawableStart=cellTrace?performance.now():0;
   const next=await prepareSceneDrawables(loaded,drawableShared,{sceneHash:asset.contentHash,
     prepareRenderable:transitions?mesh=>{
       const source=originalMesh(mesh),kind=classOf(source.name);
       const nativeTree=getLayer(mesh)==='vegetation' && /^(foliage|leaf\w*|bark|trunk|wood|frond)$/.test(kind);
       transitions.tag(mesh,cell.id,nativeTree?drawTier.detailTree:drawTier.detail);
     }:undefined,sharedFoliage:distant?.sharedFoliagePieces.has(cell.id),signal,yieldCpu:openingCells?undefined:yieldCell,
     stage:cellTrace?phase=>{const end=performance.now();cellTrace.record(cell.id,'drawables:'+phase,drawableStart,end-drawableStart);drawableStart=end;}:undefined});
   let windows:ReturnType<typeof buildFacadeMap>|undefined, shadows:ReturnType<typeof buildDepthProxies>|undefined;
   let removeSwap:(()=>void)|undefined, attached=false;
   let retirement:AbortController|undefined;
   function removeFrom<T>(list:T[],items:T[]){const removed=new Set(items);let keep=0;for(const item of list)if(!removed.has(item))list[keep++]=item;list.length=keep;}
   const release=()=>{
     retirement?.abort();handoffs.delete(cell.id);
     const disposeDone=cellTrace?.begin(cell.id,'dispose');
     if(attached){
       removeSwap?.();liveCells.delete(cell.id);fullIds.delete(cell.id);fullRevision++;cellShadows.delete(cell.id);detailWalls.delete(cell.id);
       removeFrom(glassCandidates,next.renderables);
       removeFrom(reflectiveMaterials,next.probeMaterials);
       for(const texture of next.textures)deferredTextures.delete(texture);
       if(pendingTextureWarmup.length){pendingTextureWarmup=pendingTextureWarmup.slice(textureWarmAt).filter(t=>!next.textures.has(t));textureWarmAt=0;}
       removeFrom(objects,next.objects);
       if(distant)distant.setResident(liveCells.keys());
       else if(transitions)void transitions.target(cell.id,'detail',false,true);
       else for(const proxy of hulls.get(cell.id)??[])proxy.visible=layerState.get(getLayer(proxy))!==false;
       syncStreamWindows?.();
     }
     shadows?.root.removeFromParent();shadows?.dispose();windows?.dispose();next.dispose();
     disposeDone?.();
   };
   try{
     const surfaceDone=cellTrace?.begin(cell.id,'facades-and-material-hooks');
     const materials=new Set<THREE.Material>();
     for(const mesh of next.renderables)for(const m of Array.isArray(mesh.material)?mesh.material:[mesh.material])materials.add(m);
     next.weathering?.setHeight(skyVis.height.texture,reach+30);
     if(next.glazedWindows.length && next.weathering){windows=buildFacadeMap(next.glazedWindows,reach+30);next.weathering.setFacadeMap(windows);}
     finishSurfaces(materials);occludeMaterials(materials,skyVis,bounce);streetLights.apply(materials);
     reflectStreet(next.probeMaterials,temporal,probeState);
     surfaceDone?.();const proxiesDone=cellTrace?.begin(cell.id,'depth-proxies');
     const detailMeshes=next.renderables.filter(mesh=>!next.foliageBatches.includes(mesh));
     for(const mesh of detailMeshes)if(NO_SUN_SHADOW.test(clsOf(mesh)))mesh.castShadow=false;
     shadows=openingCells?buildDepthProxies(detailMeshes,getLayer,54,isDetail,true)
       :await buildDepthProxiesAsync(detailMeshes,getLayer,54,isDetail,true,async()=>{await yieldCell();signal.throwIfAborted();});
     const shadowMeshes=[...next.renderables,...shadows.meshes];
     if(transitions)for(const mesh of shadowMeshes)if(mesh.castShadow)transitions.shadow(mesh);
     const skipped=new Set(shadows.skipped);
     const swap=staticDepthSwap({proxies:shadows.root,statics:detailMeshes.filter(m=>!skipped.has(m)),enabled:()=>budget.proxies,near:()=>camera.position,detailM:50});
     next.root.updateMatrixWorld(true);next.root.traverse(o=>{o.matrixAutoUpdate=false;});next.root.updateMatrixWorld=()=>{};
     proxiesDone?.();
     if(!openingCells){
       for(const m of next.probeMaterials)if(localReflection){m.envMap=localReflection.texture;m.envMapIntensity=m.userData.envIntensity??.7;}
       const upload=createDrawableTextureUpload(next,renderer);
       while(upload.pending){
         await yieldCell(); signal.throwIfAborted();
         const start=performance.now();
         do{upload.uploadNext(1);}while(upload.pending && performance.now()-start<1.3);
         cellTrace?.record(cell.id,'texture-submit',start);
       }
       await prewarmCell(renderer,scene,camera,scenePass.target,cellWarmTarget,shadowMeshes,
         [...warmupCompileObjects(next.root),...warmupCompileObjects(shadows!.root),...(transitions?.shadowWarmup(shadowMeshes)??[])],()=>signal.throwIfAborted(),yieldCell,
         cellTrace?(phase,start,duration)=>cellTrace.record(cell.id,phase,start,duration):undefined,{lights:stagingLights});
     }
     return {attach(){
       const attachDone=cellTrace?.begin(cell.id,'attach');
       attached=true;liveCells.set(cell.id,next);fullIds.add(cell.id);fullRevision++;cellShadows.set(cell.id,shadows!.meshes);scene.add(next.root,shadows!.root);
       const walls=new Map<string,Set<THREE.Mesh>>();
       for(const source of next.objects){
         if(!['buildings','facades'].includes(getLayer(source)))continue;
         const id=getId(source);if(!id)continue;
         let group=walls.get(id);if(!group){group=new Set();walls.set(id,group);}
         group.add(source.userData.batched||source);
       }
       detailWalls.set(cell.id,walls);
       removeSwap=depthPasses.addSwap(swap);
       objects.push(...next.objects);glassCandidates.push(...next.renderables);reflectiveMaterials.push(...next.probeMaterials);
       for(const m of next.probeMaterials)if(localReflection){m.envMap=localReflection.texture;m.envMapIntensity=m.userData.envIntensity??.7;}
       if(transitions){
         transitions.independentTree(cell.id,!!separateTreeClasses?.size || !!treeLayer?.availableCells.has(cell.id));
         void transitions.target(cell.id,'detail',true,openingCells);
       }
       else for(const proxy of hulls.get(cell.id)??[])proxy.visible=false;
       distant?.setResident(liveCells.keys());
       for(const mesh of next.renderables){
         if(layerState.get(getLayer(mesh))===false)mesh.visible=false;
         if(streamEvidence)mesh.material=cachedEvidenceFor(mesh);
       }
       for(const mesh of shadows!.meshes)if(layerState.get(getLayer(mesh))===false)mesh.visible=false;
       syncStreamWindows?.();
       attachDone?.();
     },retire:transitions?async()=>{
       const request=retirement=new AbortController();
       fullIds.delete(cell.id);handoffs.add(cell.id);fullRevision++;
       // A distant or offscreen cell can fade to its permanent exterior. For
       // an on-screen middle facade, keep full coverage through GPU preparation.
       if(middle){
         updateScenery(performance.now());
         const ready=await middle.waitForPresentation(cell.id,request.signal);
         if(request.signal.aborted)return;
         if(ready)await transitions.target(cell.id,'facade',true,transitions.value(cell.id,'detail')===1);
       }
       if(!request.signal.aborted)await transitions.target(cell.id,'detail',false);
     }:undefined,
       restore:transitions?()=>{retirement?.abort();handoffs.delete(cell.id);fullIds.add(cell.id);fullRevision++;void transitions.target(cell.id,'detail',true);}:undefined,dispose:release};
   }catch(error){release();throw error;}
 }):null;
 if(stream && drivePath && streamIndex){
   const pos=car?.group.visible?car.position():[camera.position.x,camera.position.z];
   updateDriveSight();stream.update(drivePath.project(pos[0],pos[1]),pos[0],-pos[1],0,false,driveSight);await stream.waitForCurrent();openingCells=false;updateScenery(performance.now());
 }
 onProgress('Preparing shaders and textures…');await new Promise(r=>setTimeout(r,0));
 {applySize();contact?.(mode==='car'&&!!car?.group.visible);streetLights.headlights(mode==='car'&&car?car.group:null);atmosphere.update(performance.now());patchUniforms();
  // The first reflection needs the street shadows. A small unlit pass fills
  // those full-resolution maps without compiling the temporary sky-lit street.
  const shadowPrimer = new THREE.MeshBasicMaterial({colorWrite:false, depthWrite:false});
  const primerTarget = new THREE.WebGLRenderTarget(1,1);
  const originalMaterial = scene.overrideMaterial;
  const previousTarget = renderer.getRenderTarget();
  try {
   scene.overrideMaterial = shadowPrimer;
   renderer.setRenderTarget(primerTarget);
   renderer.render(scene, camera);
  } finally {
   scene.overrideMaterial = originalMaterial;
   renderer.setRenderTarget(previousTarget);
   primerTarget.dispose();
   shadowPrimer.dispose();
  }

 stage('shadow');
  updateReflections(performance.now(),true);reviewLightReset=false;
  // Compile against the real street probe, with the actual mesh/attribute
  // layouts. Only the rendered hierarchy participates; hidden picking sources
  // would otherwise compile thousands of unused material copies.
  contact?.(mode==='car'&&!!car?.group.visible);streetLights.headlights(mode==='car'&&car?car.group:null);
 stage('reflection');
  const compileView=new THREE.Group();
  const carVisible=car?.group.visible??false;
  if(car)car.group.visible=true;
  const compileObjects=warmupCompileObjects(scene);
  compileView.traverse=callback=>compileObjects.forEach(callback);
  await renderer.compileAsync(compileView,camera,scene);stage('compile');
  // Upload nearby textures and buffers while loading is still visible. Distant
  // textures are preloaded only as exploration approaches; no whole-city uncull.
  const warmOrigin=camera.position.clone(),nearWarmM=240;
  const nearTextures=new Set<THREE.Texture>();
  camera.updateMatrixWorld();const openingFrustum=new THREE.Frustum().setFromProjectionMatrix(new THREE.Matrix4().multiplyMatrices(camera.projectionMatrix,camera.matrixWorldInverse));
  const carMeshes=new Set<THREE.Object3D>();car?.group.traverse(object=>carMeshes.add(object));
  const hidden:THREE.Object3D[]=[],culled:THREE.Object3D[]=[];
  const worldSphere=new THREE.Sphere();
  scene.traverseVisible(object=>{
   const mesh=object as THREE.Mesh;if(!mesh.geometry)return;
   if(!mesh.geometry.boundingSphere)mesh.geometry.computeBoundingSphere();
   worldSphere.copy(mesh.geometry.boundingSphere!).applyMatrix4(mesh.matrixWorld);
   const distance=worldSphere.distanceToPoint(warmOrigin);
   const near=distance<=nearWarmM||openingFrustum.intersectsSphere(worldSphere)||carMeshes.has(object)||object.children.length>0;
   for(const material of Array.isArray(mesh.material)?mesh.material:[mesh.material])for(const value of Object.values(material))if(value instanceof THREE.Texture){if(near)nearTextures.add(value);else{const spheres=deferredTextures.get(value)??[];spheres.push(worldSphere.clone());deferredTextures.set(value,spheres);}}
   if(!near){object.visible=false;hidden.push(object);}
   else if(object.frustumCulled){object.frustumCulled=false;culled.push(object);}
  });
  for(const texture of nearTextures)deferredTextures.delete(texture);
  const warm=new THREE.WebGLRenderTarget(80,48,{type:THREE.HalfFloatType,samples:2});
  const saved=renderer.getRenderTarget(),autoShadow=renderer.shadowMap.autoUpdate;
  try{renderer.shadowMap.autoUpdate=false;renderer.setRenderTarget(warm);withWarmupPrimitives(scene,()=>renderer.render(scene,camera));}
  finally{renderer.setRenderTarget(saved);renderer.shadowMap.autoUpdate=autoShadow;warm.dispose();hidden.forEach(o=>{o.visible=true;});culled.forEach(o=>{o.frustumCulled=true;});if(car)car.group.visible=carVisible;}
  // The first six-face capture already filtered the opening reflection.
  // Draw the real full-size post chain once before handing off driving.
  temporal.track(mode==='car'&&car?car.group:null);contact?.(mode==='car'&&!!car?.group.visible);streetLights.headlights(mode==='car'&&car?car.group:null);carLod?.update(camera);glass.prepare();composer.render(0);
 }
 stage('warm-frame');
 (window as unknown as {__car?:unknown}).__car={car,field,ground:carGround,objects,getLayer};
 let frame=0,previous=performance.now();const forward=new THREE.Vector3(),right=new THREE.Vector3(),worldUp=new THREE.Vector3(0,1,0),delta=new THREE.Vector3();
 let frameMs=16,lastCalls=0,lastTriangles=0;renderer.info.autoReset=false;renderer.info.reset();
 // Optional in-game readout. Count completed game frames, include hitches, and
 // show the actual scene target even when adaptive quality changes its size.
 const meter=new URLSearchParams(location.search).get('fps')==='1'?document.createElement('output'):null;
 let meterStart=previous,meterFrames=0,meterWorst=0,meterReset=false;
 const resetMeter=()=>{meterReset=true;resampleBudget=true;};
 document.addEventListener('visibilitychange',resetMeter);
 if(meter){meter.className='drive-frame-meter';meter.setAttribute('aria-live','off');host.appendChild(meter);}

 // Adaptive quality, for 60 frames a second. Two signals: frames that missed a 60 Hz deadline (an interval over 20 ms),
 // and the time the frame's own work took. The browser holds the frame's WebGL calls back when the GPU is behind, so
 // that time stays near the frame's pure CPU cost (a few ms) while there is headroom and climbs toward the whole frame
 // when there is none: it shows headroom even when the display's refresh caps the frame rate. Missed deadlines or
 // saturated work first thin the grass blades (to 70 % per metre), then step the render scale down a level (12 % fewer
 // pixels each), then switch off the ambient occlusion, the sun's light shafts and the bloom, then thin the grass again
 // (45 %); sustained headroom brings them back in the opposite order. A step up that soon has to be undone bars
 // further steps up for 20 s. Every step reallocates the render targets, so steps are at least 1.5 s apart.
 const LEVELS=[1,.94,.88,.82,.76,.7];let level=0,workMs=6,lastWork=6,lastChange=performance.now(),upBlockedUntil=0,lastUp=-1e9;const late=new Uint8Array(90);let lateCount=0,lateAt=0,settle=60;   // the first second after load is no measure
 const scaleAt=(l:number)=>Math.max(Math.min(.5,maxScale),maxScale*LEVELS[l]);
 let stillSince=0,grassShare=1;
 const thinGrass=(share:number)=>{grassShare=share;grass?.density(share);};
 const fullQuality=()=>{ao.enabled=true;grade.bloom.enabled=true;if(sunEffects)sunEffects.shafts=true;if(grassShare<1)thinGrass(1);};
 function adapt(now:number,raw:number){
  if(qualityLocked||document.hidden){resampleBudget=true;return;}
  if(temporal.stationary){if(!stillSince)stillSince=now;}else stillSince=0;
  const parked=stillSince>0 && now-stillSince>700;
  // Once the view settles, restore the display's bounded native resolution in
  // one resize. Expensive lighting can still adapt while inspecting the street.
  if(parked && level>0){
   level=0;renderScale=maxScale;resizePending=true;resampleBudget=true;return;
  }
  if(resampleBudget){resampleBudget=false;late.fill(0);lateCount=0;lateAt=0;settle=45;lastChange=now;return;}                       // a hidden or throttled tab is no measure
  if(settle>0){settle--;return;}                                         // the frames after a step reallocate targets
  const miss=raw>20?1:0;lateCount+=miss-late[lateAt];late[lateAt]=miss;lateAt=(lateAt+1)%late.length;
  workMs+=(lastWork-workMs)*.08;
  if(now-lastChange<1500)return;
  const missRate=lateCount/late.length;
  const step=(down:boolean)=>{lastChange=now;settle=10;late.fill(0);lateCount=0;workMs=down?8:6;
   if(down){if(now-lastUp<3000)upBlockedUntil=now+20000;
    if(grass&&grassShare===1)thinGrass(.7);else if(!parked && level<LEVELS.length-1){level++;renderScale=scaleAt(level);resizePending=true;}else if(ao.enabled)ao.enabled=false;else if(sunEffects?.shafts)sunEffects.shafts=false;else if(grade.bloom.enabled)grade.bloom.enabled=false;else if(grass&&grassShare>.45)thinGrass(.45);}
   else{lastUp=now;if(grassShare<.7)thinGrass(.7);else if(!grade.bloom.enabled)grade.bloom.enabled=true;else if(sunEffects&&!sunEffects.shafts)sunEffects.shafts=true;else if(!ao.enabled)ao.enabled=true;else if(level>0){level--;renderScale=scaleAt(level);resizePending=true;}else if(grassShare<1)thinGrass(1);}};
  // Keep native detail while the display is meeting its deadline. A busy CPU
  // with on-time frames is not a reason to lose street detail.
  if(missRate>.06||(missRate>.015&&workMs>16.6))step(true);
  else if(now-lastChange>4000&&now>upBlockedUntil&&missRate<.012&&workMs<12.5&&(level>0||!ao.enabled||!grade.bloom.enabled||grassShare<1||sunEffects?.shafts===false))step(false);
 }
 function animate(now:number){firstFramePending=false;frame=requestAnimationFrame(animate);const raw=Math.max(0,now-previous);const dt=Math.min(raw/1000,.05);previous=now;const workStart=performance.now();
  if(!document.hidden&&!meterReset)frameMs+=(raw-frameMs)*.06;adapt(now,raw);if(resizePending)applySize();
  if(mode==='block-replay'&&path){
   if(blockPlaying){blockDistance=Math.min(path.length,blockDistance+dt*10);if(blockDistance>=path.length)blockPlaying=false;}
   placeReplayCamera();
  }else if(mode==='street'){
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
   const len=Math.hypot(camera.position.x,camera.position.z),limit=Math.max(1,(data.contextRadiusM??studyR)-2);if(len>limit){camera.position.x*=limit/len;camera.position.z*=limit/len;}camera.position.y=groundAt(camera.position.x,camera.position.z)+1.25;
  }else if(mode==='car'&&car){
   const input:CarInput={throttle:(keys.has('KeyW')||keys.has('ArrowUp'))?1:0,brake:(keys.has('KeyS')||keys.has('ArrowDown'))?1:0,steer:((keys.has('KeyA')||keys.has('ArrowLeft'))?1:0)-((keys.has('KeyD')||keys.has('ArrowRight'))?1:0),handbrake:keys.has('Space')};
   car.update(dt,input);if(!drag)car.relax(dt);
  }else{if(targetPosition&&targetLook){camera.position.lerp(targetPosition,.07);controls.target.lerp(targetLook,.07);if(camera.position.distanceTo(targetPosition)<.05){targetPosition=null;targetLook=null;}}controls.update();}
  if(stream && drivePath){
    const e=mode==='car'&&car?car.group.position.x:camera.position.x;
    const z=mode==='car'&&car?car.group.position.z:camera.position.z;
    updateDriveSight();
    stream.update(drivePath.project(e,z),e,-z,mode==='car'&&car?Math.abs(car.forwardSpeedMps):0,true,driveSight);
    updateScenery(now);
  }
  sceneryTime=advancePresentationTime(sceneryTime,raw);transitions?.advance(sceneryTime);
  atmosphere.update(now);horizon?.update(now);grass?.update(camera.position,now/1000);foliageWind.value=now/1000;
  contact?.(mode==='car'&&!!car?.group.visible);streetLights.headlights(mode==='car'&&car?car.group:null);
  temporal.track(mode==='car'&&car?car.group:null);temporal.reflections=qualityLocked||level<LEVELS.length-1;
  grade.motion(camera,mode==='car'&&car?car.group:null,mode==='car'&&car?car.telemetry().speedKmh:0,raw/1000);
  updateReflections(now,reviewLightReset);reviewLightReset=false;
  comparison?.update(camera);carLod?.update(camera);glass.prepare();patchUniforms();composer.render(dt);stagePressure.frame(stagingQueue.length>0);
  // Count uploads/staging draws between frames as well as reflections and the main view.
  lastCalls=renderer.info.render.calls;lastTriangles=renderer.info.render.triangles;renderer.info.reset();
  // Refresh the preload neighborhood after meaningful travel, not every frame.
  // Keep an 80 m lead over startup's radius and leave distant images untouched.
  if(deferredTextures.size&&camera.position.distanceToSquared(textureWarmOrigin)>40*40){
   textureWarmOrigin.copy(camera.position);pendingTextureWarmup=nearbyTextureWarmup(deferredTextures,camera.position,320);textureWarmAt=0;
  }
  // A single upload can exceed a nominal time slice. Submit at most one, and
  // only when this frame has spare CPU time; never add work to an overloaded frame.
  if(textureWarmAt<pendingTextureWarmup.length&&document.visibilityState!=='hidden'&&raw<20&&performance.now()-workStart<10){
   const texture=pendingTextureWarmup[textureWarmAt++];renderer.initTexture(texture);deferredTextures.delete(texture);
   if(textureWarmAt===pendingTextureWarmup.length){pendingTextureWarmup=[];textureWarmAt=0;}
  }
  lastWork=performance.now()-workStart;
  if(!meter)meterReset=false;
  if(meter){
   if(document.hidden||meterReset){meterReset=false;meterStart=now;meterFrames=0;meterWorst=0;}
   else{
    meterFrames++;meterWorst=Math.max(meterWorst,raw);
    if(now-meterStart>=1000){
     meter.textContent=`${Math.round(1000*meterFrames/(now-meterStart))} FPS · ${sceneW}×${sceneH} · longest frame ${Math.round(meterWorst)} ms`;
     meterStart=now;meterFrames=0;meterWorst=0;
    }
   }
  }
 }
 visibilityReady=true;frame=requestAnimationFrame(animate);
 // Review hook: `pose` puts the orbit camera at metric ENU coordinates (as ?viewFrom/?viewAt do) without a reload,
 // `capture` renders one frame and returns it as a PNG blob; both are for same-pose A/B comparisons only. `pose` lifts
 // the orbit's ground limit, which would otherwise raise a street-level camera looking up at a facade to its target's height.
 const coverage=()=>({
   distant:distant?.stats??null,
   active:[...liveCells.keys()],
   shownDistant:[...new Set(farMeshes.filter(m=>m.visible&&!m.userData.persistentFoliage).map(m=>String(m.userData.streamCell)))],
   visibleCollisionHulls:[...hulls.values()].reduce((n,meshes)=>n+meshes.filter(m=>m.visible).length,0),
 });
 function frontageCoverage(){
   const owners=[];
   const shown=(mesh:THREE.Mesh)=>mesh.visible&&mesh.geometry.drawRange.count!==0&&
     (mesh.userData.submissionController||!!mesh.parent)&&
     (Array.isArray(mesh.material)?mesh.material.some(m=>m.visible):mesh.material.visible);
   for(const [cell,entry] of frontageBounds){
     if(entry.near>1)continue;
     const resident=liveCells.has(cell.id),walls=detailWalls.get(cell.id);
     const detailCover=transitions?.value(cell.id,'detail')??(resident?1:0);
     const middleCover=middleGroups.get(cell.id)?.visible?(1-detailCover)*(transitions?.value(cell.id,'facade')??0):0;
     const middleVisible=middleCover>0;
     const exteriorVisible=(hulls.get(cell.id)??[]).some(mesh=>getLayer(originalMesh(mesh))==='facades'&&shown(mesh));
     for(const feature of entry.owners){
       if(feature.revision!==driveSight.revision||feature.priority>1)continue;
       const meshes=walls?.get(feature.id);
       const detailVisible=detailCover>0&&!!meshes&&[...meshes].some(shown);
       owners.push({id:feature.id,cell:cell.id,nearM:feature.distance,resident,detailVisible,middleVisible,exteriorVisible,
         windowCoverage:(detailVisible?detailCover:0)+middleCover});
     }
   }
   let correctedExteriorFaces=0;
   for(const detail of liveCells.values())correctedExteriorFaces+=detail.model.scene.userData.correctedExteriorFaces??0;
   return {owners,correctedExteriorFaces};
 }
 const reviewView={grass:()=>grass,sunEffects,stream,scenery:()=>({facades:middle?.status(),trees:treeLayer?.stats,transitions:transitions?.status(),errors:sceneryErrors}),coverage,frontageCoverage,streamTimings:()=>cellTrace?.read()??[],imagePool:()=>imagePool?.stats(),quality:(scale?:number,lock=true)=>{qualityLocked=lock;if(scale){renderScale=scale;level=LEVELS.reduce((best,_,i)=>Math.abs(scaleAt(i)-scale)<Math.abs(scaleAt(best)-scale)?i:best,0);resizePending=true;}return {renderScale,maxScale,level,workMs:+workMs.toFixed(1),ao:ao.enabled,bloom:grade.bloom.enabled,shafts:sunEffects?.shafts??false,grass:grassShare};},exposure:()=>grade.exposure.read(renderer),pose:(from:number[],at:number[])=>{grade.snap();temporal.reset();mode='orbit';controls.enabled=true;controls.maxPolarAngle=Math.PI*.98;targetPosition=null;targetLook=null;highlight.visible=false;camera.position.set(from[0],from[2],-from[1]);controls.target.set(at[0],at[2],-at[1]);controls.update();},
  capture:(w=1800,h=1125)=>new Promise<Blob|null>(resolve=>{qualityLocked=true;fullQuality();grade.snap();temporal.reset();
   renderer.setDrawingBufferSize(w,h,1);composer.setSize(w,h);outputW=sceneW=0;camera.aspect=w/h;camera.updateProjectionMatrix();atmosphere.resize();
   // the animation loop may be throttled (hidden pane): place the shadow cascades and the reflection probe for this pose
   if(controls.enabled)controls.update();contact?.(mode==='car'&&!!car?.group.visible);streetLights.headlights(mode==='car'&&car?car.group:null);atmosphere.update(performance.now()+1000);atmosphere.csm.lights[1].shadow.needsUpdate=true;updateReflections(performance.now()+1e7,true);
   comparison?.update(camera);carLod?.update(camera);glass.prepare();patchUniforms();visibilityReview=true;try{composer.render(1/60);}finally{visibilityReview=false;}renderer.domElement.toBlob(b=>{resizePending=true;resolve(b);},'image/png');}),
  lockQuality:(on:boolean)=>{qualityLocked=on;if(on)fullQuality();},
  renderOnce:()=>{const t0=performance.now();composer.render(1/60);renderer.getContext().finish();return performance.now()-t0;},budget:{flags:budget,submissions:submissions.stats,visibility:visibility.stats,glass:glass.stats,proxies:depth.meshes.length,proxySkipped:depth.skipped.length,car:carLod?.stats??null},info:()=>({calls:lastCalls,triangles:lastTriangles,frameMs:+frameMs.toFixed(1),renderScale:+renderScale.toFixed(2),ao:ao.enabled,bloom:grade.bloom.enabled,batches:batches.length,sceneTriangles:objects.reduce((n,o)=>n+(o.userData.foliageFanCentres ? o.geometry.attributes.position.count*4 : (o.geometry.index?o.geometry.index.count:o.geometry.attributes.position.count)/3),0)}),atmosphere,scene,camera,objects,renderer,composer};
 const debug=window as unknown as {__view?:typeof reviewView;__car?:{car:Car|null}};
 debug.__view=reviewView;
 function layer(name:string,on:boolean){
  temporal.reset();
  const wanted=(candidate:string)=>candidate===name||(name==='street_objects'&&candidate==='signs');
  if(name==='street_objects'){streetLights.visible(on);if(parked)parked.group.visible=on;if(overhead)overhead.group.visible=on;}
  distant?.setLayer(name,on);
  for(const mesh of renderables)if(wanted(getLayer(mesh))&&!collisionOnly(mesh)&&!((distant||hasExteriorProxy)&&hullMesh(mesh))){
    mesh.visible=on&&!(temporaryMesh(mesh)&&liveCells.has(originalMesh(mesh).userData.streamCell));
  }
  for(const mesh of depth.meshes)if(wanted(getLayer(mesh)))mesh.visible=on;
  layerState.set(name,on);if(name==='vegetation')treeLayer?.setLayer(on);if(transitions)for(const id of hulls.keys())syncCellCoverage(id);if(name==='street_objects')layerState.set('signs',on);
  for(const cell of liveCells.values())for(const mesh of cell.renderables)if(wanted(getLayer(mesh)))mesh.visible=on;
  for(const meshes of cellShadows.values())for(const mesh of meshes)if(wanted(getLayer(mesh)))mesh.visible=on;
 }
 const evidenceMaterials=new WeakMap<THREE.Mesh,THREE.Material|THREE.Material[]>();
 function evidenceMaterial(tint:string){
  let material=evidences.get(tint);
  if(!material){material=new THREE.MeshStandardMaterial({color:tint,roughness:.9});atmosphere.setupMaterial(material);evidences.set(tint,material);}
  return material;
 }
 function cachedEvidenceFor(mesh:THREE.Mesh){
  let material=evidenceMaterials.get(mesh);
  if(!material){material=evidenceFor(mesh);evidenceMaterials.set(mesh,material);}
  return material;
 }
 function evidenceFor(mesh:THREE.Mesh){
  const tint=(o:THREE.Mesh)=>evidenceTint(getLayer(o),o.userData.evidence);
  const members=mesh.userData.members as THREE.Mesh[]|undefined;
  if(!members||members.length<2)return evidenceMaterial(tint(mesh));
  const colors=members.map(tint),palette=[...new Set(colors)];
  if(palette.length===1)return evidenceMaterial(palette[0]);
  // A colour run uses the existing concatenated indices and original winding.
  // Three uses these ranges only while the mesh has an array of evidence
  // materials. Its ordinary material still submits the whole batch in one draw.
  let cursor=0,start=0;
  for(let i=0;i<members.length;i++){
   const geometry=members[i].geometry;
   cursor+=geometry.index?.count??geometry.attributes.position.count;
   if(colors[i]!==colors[i+1]){
    mesh.geometry.addGroup(start,cursor-start,palette.indexOf(colors[i]));
    start=cursor;
   }
  }
  return palette.map(evidenceMaterial);
 }
 let evidenceShown=false;
 function evidence(on:boolean){temporal.reset();
  if(evidenceShown===on)return;
  evidenceShown=on;submissions.setEnabled(!on);
  streamEvidence=on;
  for(const cell of liveCells.values())for(const mesh of cell.renderables){
    mesh.material=on?cachedEvidenceFor(mesh):(mesh.userData.original??cell.originals.get(mesh));
  }
  for(const mesh of renderables){
   if(!on){mesh.material=(mesh.userData.original??originals.get(mesh)) as THREE.Material|THREE.Material[];continue;}
   mesh.material=cachedEvidenceFor(mesh);
  }
 }
 let disposed=false;
 function disposeActiveTextures(){
  const materials=new Set<THREE.Material>([...originals.values()].flat());
  scene.traverse(object=>{const material=(object as THREE.Mesh).material;if(material)for(const m of Array.isArray(material)?material:[material])materials.add(m);});
  const textures=new Set<THREE.Texture>([leafTexture]);
  for(const material of materials)for(const value of Object.values(material))if(value instanceof THREE.Texture&&!value.isRenderTargetTexture)textures.add(value);
  disposeOwnedTextures(textures);deferredTextures.clear();pendingTextureWarmup=[];
 }
 return {view,select,lighting,damp,nextStreet,drive,car:driveCar,block,layer,evidence,dispose:()=>{if(disposed)return;disposed=true;document.removeEventListener('visibilitychange',resetMeter);meter?.remove();stream?.dispose();treeLayer?.dispose();void middle?.dispose();transitions?.dispose();stagePressure.dispose();distant?.dispose();cellWarmTarget.dispose();disposeActiveTextures();releaseParserImageSources(model.parser);bakedRoad?.dispose();bakedPaving?.dispose();facadeMap?.dispose();horizon?.dispose();if(debug.__view===reviewView)delete debug.__view;if(debug.__car?.car===car)delete debug.__car;cancelAnimationFrame(frame);visibility.dispose();submissions.dispose();streetLights.dispose();comparison?.dispose();car?.dispose();parked?.group.children.forEach(m=>(m as THREE.BatchedMesh).dispose());parked?.materials.forEach(m=>m.dispose());overhead?.group.traverse(o=>{if(o instanceof THREE.BatchedMesh)o.dispose();else if(o instanceof THREE.InstancedMesh){o.dispose();o.geometry.dispose();}else if(o instanceof THREE.Mesh)o.geometry.dispose();});overhead?.materials.forEach(m=>m.dispose());resize.disconnect();controls.dispose();renderer.domElement.removeEventListener('pointerdown',down);window.removeEventListener('pointermove',move);window.removeEventListener('pointerup',up);window.removeEventListener('keydown',keydown);window.removeEventListener('keyup',keyup);window.removeEventListener('blur',blur);window.removeEventListener('hashchange',linkedView);objects.forEach(o=>o.geometry.dispose());batches.forEach(b=>{b.geometry.dispose();b.removeFromParent();if(b.name==='vegetation___foliage'&&b===foliageBatches.find(mesh=>mesh.name===b.name)){const m=b.userData.original as THREE.MeshStandardMaterial;m.dispose();}});wxGround.dispose();new Set([...originals.values()].flat()).forEach(m=>m.dispose());evidences.forEach(m=>m.dispose());localReflection?.dispose();probeSpare?.dispose();reflectionProbe.dispose();if(reflectionFilter!==atmosphere.environmentFilter)reflectionFilter.dispose();bounce?.dispose();skyVis.dispose();depth.dispose();temporal.dispose();scenePass.dispose();disposeOcclusionResolve();ao.dispose();grass?.dispose();grade.dispose();sunEffects?.dispose();composer.dispose();atmosphere.dispose();ring.geometry.dispose();(ring.material as THREE.Material).dispose();highlight.geometry.dispose();(highlight.material as THREE.Material).dispose();renderer.dispose();renderer.domElement.remove();}};
}
