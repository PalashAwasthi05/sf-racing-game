import * as THREE from 'three';
import type {Overhead} from './overhead';
import type {GlazingEmitter} from './glazing';
import {signalHeads,signalLights} from './street-signals.ts';
import {buildWindowSpill} from './window-spill.ts';

// A lamp record is its actual lens center in world space and mounting height above pavement.
// Resolve the nearby fixtures per spatial cell once. A moving camera, a paint reflection and
// an aerial view all query the same stationary lights without adding per-fixture shadow passes.
const CELL = 16;
const RANGE = 3.6;
const TABLE_WIDTH = 256;

export function buildLampGrid(lamps: THREE.Vector4[]) {
  const bounds = new THREE.Box2();
  for (const lamp of lamps) {
    const r = lamp.w * RANGE;
    bounds.expandByPoint(new THREE.Vector2(lamp.x-r, lamp.z-r));
    bounds.expandByPoint(new THREE.Vector2(lamp.x+r, lamp.z+r));
  }
  if (bounds.isEmpty()) bounds.set(new THREE.Vector2(0,0),new THREE.Vector2(0,0));
  const size = bounds.getSize(new THREE.Vector2());
  // Bound metadata memory for very large or disconnected model areas.
  const cell = Math.max(CELL, size.x/512, size.y/512);
  const cols = Math.max(1,Math.ceil(size.x/cell)), rows = Math.max(1,Math.ceil(size.y/cell));
  const cells: THREE.Vector4[][] = Array.from({length:cols*rows},()=>[]);
  for (const lamp of lamps) {
    const r = lamp.w*RANGE, x = lamp.x-bounds.min.x, z = lamp.z-bounds.min.y;
    for (let j=Math.max(0,Math.floor((z-r)/cell));j<=Math.min(rows-1,Math.floor((z+r)/cell));j++) {
      for (let i=Math.max(0,Math.floor((x-r)/cell));i<=Math.min(cols-1,Math.floor((x+r)/cell));i++) {
        const dx=Math.max(i*cell-x,0,x-(i+1)*cell), dz=Math.max(j*cell-z,0,z-(j+1)*cell);
        if (dx*dx+dz*dz<=r*r) cells[j*cols+i].push(lamp);
      }
    }
  }
  const records = cells.length + cells.reduce((sum,c)=>sum+c.length,0);
  const height = Math.ceil(records/TABLE_WIDTH);
  const packed = new Float32Array(TABLE_WIDTH*height*4);
  let offset = cells.length;
  for (let i=0;i<cells.length;i++) {
    packed[4*i]=offset;
    packed[4*i+1]=cells[i].length;
    for (const lamp of cells[i]) lamp.toArray(packed,4*offset++);
  }
  const texture = new THREE.DataTexture(packed,TABLE_WIDTH,height,THREE.RGBAFormat,THREE.FloatType);
  texture.minFilter=texture.magFilter=THREE.NearestFilter;
  texture.generateMipmaps=false;
  texture.needsUpdate=true;
  return {texture, origin:bounds.min, shape:new THREE.Vector3(cols,rows,cell)};
}

// Older worlds batch lenses and housings into two street-object meshes. Recover their
// separate connected parts, including exports that split vertices at flat-shading seams.
function streetParts(mesh: THREE.Mesh): THREE.Box3[] {
  const geometry=mesh.geometry, position=geometry.getAttribute('position'), indices=geometry.getIndex();
  if (!position || !position.count) return [];
  mesh.updateWorldMatrix(true,false);
  const points=new Map<string,number>(), vertex=new Int32Array(position.count), parent: number[]=[];
  const p=new THREE.Vector3();
  function root(a:number) {
    while (a!==parent[a]) { parent[a]=parent[parent[a]]; a=parent[a]; }
    return a;
  }
  for (let i=0;i<position.count;i++) {
    p.fromBufferAttribute(position,i);
    const key=Math.round(p.x*1000)+','+Math.round(p.y*1000)+','+Math.round(p.z*1000);
    let id=points.get(key);
    if (id===undefined) {id=points.size;points.set(key,id);parent.push(id);}
    vertex[i]=id;
  }
  const count=indices?.count??position.count;
  for (let i=0;i+2<count;i+=3) {
    const a=root(vertex[indices?indices.getX(i):i]);
    parent[root(vertex[indices?indices.getX(i+1):i+1])]=a;
    parent[root(vertex[indices?indices.getX(i+2):i+2])]=a;
  }
  const boxes=new Map<number,THREE.Box3>();
  for (let i=0;i<position.count;i++) {
    const id=root(vertex[i]);let box=boxes.get(id);
    if (!box) {box=new THREE.Box3();boxes.set(id,box);}
    box.expandByPoint(p.fromBufferAttribute(position,i).applyMatrix4(mesh.matrixWorld));
  }
  return [...boxes.values()];
}

export function streetLampPositions(meshes: THREE.Mesh[], overhead: Overhead|null, groundAt:(x:number,z:number)=>number, parts=streetParts) {
  const lamps=[...(overhead?.lamps??[])];
  const light: THREE.Box3[]=[], housings: THREE.Box3[]=[];
  const p=new THREE.Vector3(), extent=new THREE.Vector3();
  for (const mesh of meshes) {
    const [prefix,kind]=mesh.name.split('___');
    if ((mesh.userData.layer??prefix)!=='street_objects' || (kind!=='white' && kind!=='charcoal')) continue;
    for (const box of parts(mesh)) {
      box.getSize(extent); box.getCenter(p);
      if (extent.y>.30 || Math.max(extent.x,extent.z)>2.0 || Math.min(extent.x,extent.z)<.1) continue;
      if (p.y-groundAt(p.x,p.z)<2.7) continue;
      if (kind==='white' && extent.y<.10) light.push(box);
      else if (kind==='charcoal') housings.push(box);
    }
  }
  for (const lens of light) {
    lens.getCenter(p);
    if (!housings.some(b=>b.min.y>=lens.min.y-.012 && b.min.y<lens.max.y+.15 &&
      b.min.x<=lens.min.x+.025 && b.max.x>=lens.max.x-.025 && b.min.z<=lens.min.z+.025 && b.max.z>=lens.max.z-.025)) continue;
    if (lamps.some(l=>Math.hypot(l.x-p.x,l.y-p.y,l.z-p.z)<.8)) continue;
    lamps.push(new THREE.Vector4(p.x,p.y,p.z,p.y-groundAt(p.x,p.z)));
  }
  return lamps;
}

const DECLARATIONS = /* glsl */`
uniform float uRoadNight;
uniform sampler2D uLampGrid;
uniform vec2 uLampOrigin;
uniform vec3 uLampShape;
uniform float uStreetOn;
uniform float uBeamOn;
uniform vec3 uBeamLeft, uBeamRight, uBeamForward, uBeamUp, uBeamSide;
uniform vec3 uRearLeft,uRearRight;uniform float uRearPower;
vec4 lampRecord(int i) {return texelFetch(uLampGrid,ivec2(i%256,i/256),0);}
`;

const DIRECT = /* glsl */`
if (uRoadNight > 0.5) {
  // Inverse view, including the proper instance / BatchedMesh transform. This is
  // also correct for every cube face rendered into a material reflection probe.
  vec3 roadP=cameraPosition-vViewPosition*mat3(viewMatrix);
  /* window irradiance */
  ivec2 cell=ivec2(floor((roadP.xz-uLampOrigin)/uLampShape.z));
  if(uStreetOn>.5 && all(greaterThanEqual(cell,ivec2(0))) && all(lessThan(cell,ivec2(uLampShape.xy)))) {
    vec4 span=lampRecord(cell.y*int(uLampShape.x)+cell.x);
    for(int i=0;i<int(span.y);i++){
      vec4 lamp=lampRecord(int(span.x)+i);
      vec3 delta=lamp.xyz-roadP;
      float d2=max(dot(delta,delta),.16), invD=inversesqrt(d2);
      float cosine=delta.y*invD;
      float falloff=1.-smoothstep(.62,1.,d2/(lamp.w*lamp.w*12.96));
      if(cosine>0. && falloff>.0) {
        // Shielded street luminaire. The distribution spreads light towards its
        // sides, with a cutoff above the lens. Height sets the extent of its pool.
        IncidentLight lampLight;
        lampLight.direction=mat3(viewMatrix)*(delta*invD);
        lampLight.visible=true;
        float distribution=smoothstep(.015,.40,cosine)*(1.+.95*(1.-cosine*cosine));
        lampLight.color=vec3(1.,.71,.44)*(3.9*lamp.w*lamp.w/max(d2,1.0)*distribution*falloff);
        #ifdef HAS_CAR_CONTACT
        lampLight.color*=carLampVisibility(lamp.xyz,roadP);
        #endif
        RE_Direct(lampLight,geometryPosition,geometryNormal,geometryViewDir,geometryClearcoatNormal,material,reflectedLight);
      }
    }
  }
  if(uBeamOn>.5){
    for(int i=0;i<2;i++){
      vec3 delta=roadP-(i==0?uBeamLeft:uBeamRight);
      float ahead=dot(delta,uBeamForward);
      if(ahead>.15 && ahead<85.){
        float h=dot(delta,uBeamSide)/ahead, v=dot(delta,uBeamUp)/ahead;
        // Low beams place most flux just below the cutoff. A broad foreground
        // wash becomes two long, overlapping lobes, gently splayed outwards.
        float aim=i==0?-.035:.035;
        float cutoff=1.-smoothstep(-.012,.009,v);
        float side=abs(h-aim);
        float lobe=exp(-24.*side*side-112.*(v+.032)*(v+.032));
        float foreground=.21*exp(-6.*side*side-19.*(v+.11)*(v+.11));
        float d2=dot(delta,delta);
        float beam=(lobe+foreground)*cutoff*(1.-smoothstep(50.,85.,ahead));
        if(beam>.001){
          IncidentLight headlight;
          headlight.direction=mat3(viewMatrix)*(-delta*inversesqrt(max(d2,.04)));
          headlight.color=vec3(.88,.93,1.)*(2200.*beam/(d2+20.));
          headlight.visible=true;
          RE_Direct(headlight,geometryPosition,geometryNormal,geometryViewDir,geometryClearcoatNormal,material,reflectedLight);
        }
      }
    }
  }
}
`;

function luminousLenses(lamps: THREE.Vector4[]) {
  const geometry=new THREE.InstancedBufferGeometry();
  geometry.setAttribute('position',new THREE.Float32BufferAttribute([-1,-1,0,1,-1,0,1,1,0,-1,1,0],3));
  geometry.setIndex([0,1,2,0,2,3]);
  const xyz=new Float32Array(lamps.length*3);
  for(let i=0;i<lamps.length;i++){const p=lamps[i];xyz.set([p.x,p.y-.065,p.z],3*i);}
  geometry.setAttribute('lampCenter',new THREE.InstancedBufferAttribute(xyz,3));geometry.instanceCount=lamps.length;
  const material=new THREE.ShaderMaterial({
    depthWrite:false,transparent:true,blending:THREE.CustomBlending,
    blendSrc:THREE.OneFactor,blendDst:THREE.OneFactor,blendSrcAlpha:THREE.ZeroFactor,blendDstAlpha:THREE.OneFactor,
    vertexShader:`
      attribute vec3 lampCenter; varying vec2 lampUV; varying float lensVisibility;
      void main(){
        vec4 mv=viewMatrix*vec4(lampCenter,1.);
        vec3 eye=cameraPosition-lampCenter;
        lensVisibility=1.-smoothstep(-.05,.12,eye.y/max(length(eye),.01));
        lampUV=position.xy;
        mv.xy+=position.xy*.68;
        gl_Position=projectionMatrix*mv;
      }`,
    fragmentShader:`
      varying vec2 lampUV; varying float lensVisibility;
      void main(){
        float r2=dot(lampUV,lampUV);
        float core=2.8*exp(-r2*39.);
        float halo=.17*exp(-r2*5.5)*max(0.,1.-r2);
        gl_FragColor=vec4(lensVisibility*vec3(1.,.76,.5)*(core+halo),0.);
      }`,
  });
  const mesh=new THREE.Mesh(geometry,material);
  mesh.frustumCulled=false;mesh.name='street-lamp-luminance';mesh.renderOrder=2;
  return mesh;
}

function vehicleLamps(body: THREE.Object3D) {
  const inverse=body.matrixWorld.clone().invert();
  const front:THREE.Vector3[]=[],rear:THREE.Vector3[]=[];
  let running:THREE.MeshStandardMaterial|undefined;
  body.traverse(o=>{
    if(!(o instanceof THREE.Mesh))return;
    if(/^taillamp__tail_led(?:_\d+)?$/i.test(o.name) && o.material instanceof THREE.MeshStandardMaterial)running=o.material;
    const dest=/^headlamp_(?:left|right)__lens(?:_\d+)?$/i.test(o.name)?front:
      /^taillamp_(?:left|right)__tail_lens(?:_\d+)?$/i.test(o.name)?rear:null;
    if(!dest)return;
    o.geometry.computeBoundingBox();
    const p=o.geometry.boundingBox!.getCenter(new THREE.Vector3());
    dest.push(p.applyMatrix4(o.matrixWorld).applyMatrix4(inverse));
  });
  for(const pair of [front,rear])pair.sort((a,b)=>a.z-b.z);
  return {front,rear,running};
}
const REAR_LIGHT = /* glsl */`
 if(uRearPower>0. && dot(roadP-uRearLeft,roadP-uRearLeft)<49.){
  for(int i=0;i<2;i++){
   vec3 delta=(i==0?uRearLeft:uRearRight)-roadP;
   float d2=dot(delta,delta);
   vec3 L=delta*inversesqrt(max(.005,d2));
   float outward=smoothstep(-.04,.52,dot(L,uBeamForward));
   if(outward>0.){
    IncidentLight tail;tail.visible=true;tail.direction=mat3(viewMatrix)*L;
    tail.color=vec3(1.,.028,.011)*uRearPower*(outward/(.26+d2))*(1.-smoothstep(20.,43.,d2));
    RE_Direct(tail,geometryPosition,geometryNormal,geometryViewDir,geometryClearcoatNormal,material,reflectedLight);
   }
  }
 }`;

export function createStreetLights(meshes:THREE.Mesh[], overhead:Overhead|null, groundAt:(x:number,z:number)=>number) {
  const cachedParts=new Map<THREE.Mesh,THREE.Box3[]>();
  const parts=(mesh:THREE.Mesh)=>{let boxes=cachedParts.get(mesh);if(!boxes){boxes=streetParts(mesh);cachedParts.set(mesh,boxes);}return boxes;};
  const lamps=streetLampPositions(meshes,overhead,groundAt,parts), grid=buildLampGrid(lamps);
  const group=new THREE.Group();group.name='street-light-emission';group.visible=false;
  const glow=luminousLenses(lamps),signals=signalLights(signalHeads(meshes,parts));group.add(glow,signals);
  cachedParts.clear();
  const spillMaterials=new Set<THREE.Material>();
  const groundLayers=new Set(['roads','cycleways','transit','markings','sidewalks','paths','landscape','terrain','steps']);
  for(const mesh of meshes)if(groundLayers.has(mesh.userData.layer??mesh.name.split('___')[0])) {
    for(const m of Array.isArray(mesh.material)?mesh.material:[mesh.material])spillMaterials.add(m);
  }
  let spill=buildWindowSpill([],groundAt);
  const uniforms={
    uRoadNight:{value:0},uStreetOn:{value:1},uLampGrid:{value:grid.texture},
    uLampOrigin:{value:grid.origin},uLampShape:{value:grid.shape},
    uRearLeft:{value:new THREE.Vector3()},uRearRight:{value:new THREE.Vector3()},uRearPower:{value:0},uBeamOn:{value:0},uBeamLeft:{value:new THREE.Vector3()},uBeamRight:{value:new THREE.Vector3()},
    uBeamForward:{value:new THREE.Vector3()},uBeamUp:{value:new THREE.Vector3()},uBeamSide:{value:new THREE.Vector3()},
  };
  const spillUniforms={uWindowSpill:{value:spill.texture},uSpillBounds:{value:spill.bounds},uHasSpill:{value:0}};
  const dayIntensity=overhead?.lens.emissiveIntensity??0;
  const patched=new WeakSet<THREE.MeshStandardMaterial>();
  let lastCar:THREE.Object3D|null=null,lights:ReturnType<typeof vehicleLamps>|null=null;
  function sync() {
    glow.visible=!!(uniforms.uRoadNight.value && lamps.length);
    signals.visible=signals.geometry.instanceCount>0;
    group.visible=!!uniforms.uStreetOn.value && (glow.visible||signals.visible);
    if(overhead) overhead.lens.emissiveIntensity=group.visible&&glow.visible?5.5:dayIntensity;
  }
  sync();
  return {group,
    apply(materials: Iterable<THREE.Material>) {
      for(const mat of materials) {
        if(!(mat instanceof THREE.MeshStandardMaterial) || patched.has(mat))continue;
        patched.add(mat);
        const prev=mat.onBeforeCompile.bind(mat), cacheKey=mat.customProgramCacheKey.bind(mat);
        const receivesSpill=spillMaterials.has(mat)||mat.name==='paint';
        mat.onBeforeCompile=(shader,renderer)=>{
          prev(shader,renderer);
          Object.assign(shader.uniforms,uniforms);
          shader.fragmentShader=shader.fragmentShader
            .replace('#include <common>','#include <common>\n'+DECLARATIONS)
            .replace('#include <lights_fragment_end>','#include <lights_fragment_end>\n'+DIRECT);
          if(receivesSpill){
            Object.assign(shader.uniforms,spillUniforms);
            shader.fragmentShader=shader.fragmentShader
              .replace('#include <common>','#include <common>\nuniform sampler2D uWindowSpill;uniform vec4 uSpillBounds;uniform float uHasSpill;')
              .replace('/* window irradiance */',`
              ${mat.name==='paint'?'':REAR_LIGHT}
              if(uHasSpill>.5){
                vec2 uv=(roadP.xz-uSpillBounds.xy)*uSpillBounds.zw;
                if(all(greaterThan(uv,vec2(0.))) && all(lessThan(uv,vec2(1.)))){
                  vec4 spill=texture2D(uWindowSpill,uv);
                  if(spill.r>.0001){
                    float above=roadP.y-spill.a/spill.r;
                    float up=dot(geometryNormal,viewMatrix[1].xyz);
                    float receive=smoothstep(.05,.85,up)*(1.-smoothstep(.45,2.6,abs(above)));
                    reflectedLight.indirectDiffuse+=spill.rgb*(receive*RECIPROCAL_PI)*material.diffuseColor;
                  }
                }
              }`);
          }
        };
        mat.customProgramCacheKey=()=>cacheKey()+'|street-luminaires-v2'+(receivesSpill?'|sill':'');
        mat.needsUpdate=true;
      }
    },
    windows(sources:GlazingEmitter[],field?:Parameters<typeof buildWindowSpill>[2]){
      const next=buildWindowSpill(sources,groundAt,field);spill.texture.dispose();spill=next;
      spillUniforms.uWindowSpill.value=spill.texture;spillUniforms.uSpillBounds.value=spill.bounds;spillUniforms.uHasSpill.value=spill.count?1:0;
    },
    night(on:boolean){uniforms.uRoadNight.value=on?1:0;if(!on)uniforms.uBeamOn.value=0;sync();},
    visible(on:boolean){uniforms.uStreetOn.value=on?1:0;sync();},
    headlights(body:THREE.Object3D|null) {
      uniforms.uBeamOn.value=0;uniforms.uRearPower.value=0;
      if(!body?.visible || !uniforms.uRoadNight.value)return;
      if(lastCar!==body) {
        body.updateWorldMatrix(true,true);
        lights=vehicleLamps(body);lastCar=body;
      } else body.updateWorldMatrix(true,false);
      if(!lights)return;
      uniforms.uBeamForward.value.setFromMatrixColumn(body.matrixWorld,0).normalize();
      if(lights.rear.length>=2 && lights.running){
        uniforms.uRearLeft.value.copy(lights.rear[0]).applyMatrix4(body.matrixWorld);
        uniforms.uRearRight.value.copy(lights.rear[lights.rear.length-1]).applyMatrix4(body.matrixWorld);
        uniforms.uRearPower.value=lights.running.emissiveIntensity*1.35;
      }
      const lenses=lights.front;
      if(lenses.length<2)return;
      uniforms.uBeamLeft.value.copy(lenses[0]).applyMatrix4(body.matrixWorld);
      uniforms.uBeamRight.value.copy(lenses[lenses.length-1]).applyMatrix4(body.matrixWorld);
      uniforms.uBeamUp.value.setFromMatrixColumn(body.matrixWorld,1).normalize();
      uniforms.uBeamSide.value.setFromMatrixColumn(body.matrixWorld,2).normalize();
      uniforms.uBeamOn.value=1;
    },
    dispose(){group.removeFromParent();for(const m of [glow,signals]){m.geometry.dispose();m.material.dispose();}grid.texture.dispose();spill.texture.dispose();},
  };
}
