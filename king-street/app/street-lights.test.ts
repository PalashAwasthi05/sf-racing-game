import test from 'node:test';
import assert from 'node:assert/strict';
import * as THREE from 'three';
import {buildLampGrid,createStreetLights,streetLampPositions} from './street-lights.ts';
import type {GlazingEmitter} from './glazing';
import {buildWindowSpill} from './window-spill.ts';

test('adjacent grid cells include every contributing lamp across their shared border',()=>{
  const lamps=[new THREE.Vector4(-22,6,-3,8),new THREE.Vector4(10,4,9,6),new THREE.Vector4(-4,1,11,3)];
  const {texture,origin,shape}=buildLampGrid(lamps), values=texture.image.data as Float32Array;
  for(let x=-49;x<31;x+=.625)for(let z=-30;z<31;z+=.875){
    const i=Math.floor((x-origin.x)/shape.z),j=Math.floor((z-origin.y)/shape.z);
    const covered=lamps.filter(l=>Math.hypot(x-l.x,z-l.z)<3.6*l.w);
    if(!covered.length)continue;
    assert.ok(i>=0&&i<shape.x&&j>=0&&j<shape.y);
    const header=(j*shape.x+i)*4, start=values[header],count=values[header+1];
    const a=Array.from({length:count},(_,k)=>values[(start+k)*4]);
    for(const l of covered)assert.ok(a.includes(l.x));
  }
  texture.dispose();
});

test('recognizes luminaires from the co-located lens and housing, and excludes bare posts',()=>{
  const make=(name:string,x:number,y:number,w:number,h:number,d:number)=>{
    const mesh=new THREE.Mesh(new THREE.BoxGeometry(w,h,d));mesh.name='street_objects___'+name;
    mesh.position.set(x,y,-2);mesh.rotation.y=.47;return mesh;
  };
  const glass=make('white',0,8,.62,.025,.24),head=make('charcoal',0,8.07,.85,.12,.32);
  const bad=make('white',8,8,.62,.025,.24), post=make('charcoal',8,7.5,.1,1,.1);
  assert.deepEqual(streetLampPositions([glass,head,bad,post],null,()=>0).map(p=>p.toArray().map(Math.round)),[[0,8,-2,8]]);
});

test('low beams attach to real lens geometry and operate independently from the street-equipment layer',()=>{
  const system=createStreetLights([],null,()=>0), material=new THREE.MeshPhysicalMaterial();
  system.apply([material]);
  const shader={uniforms:{},vertexShader:'',fragmentShader:'#include <common>\n#include <lights_fragment_end>'};
  material.onBeforeCompile(shader as never,{} as THREE.WebGLRenderer);
  const u=shader.uniforms as Record<string,{value:unknown}>;
  const body=new THREE.Group();body.position.set(-3,8,11);body.rotation.y=Math.PI/2;
  for(const [name,z] of [['left',-.75],['right',.75]] as const){
    const m=new THREE.Mesh(new THREE.BoxGeometry(.1,.1,.2));m.name='headlamp_'+name+'__lens';
    m.position.set(1.7,.6,z);body.add(m);
  }
  const emitter=new THREE.MeshStandardMaterial({emissive:0xff0904,emissiveIntensity:1.4});
  const strips=new THREE.Mesh(new THREE.BoxGeometry(.1,.1,.2),emitter);strips.name='taillamp__tail_led';body.add(strips);
  for(const [name,z] of [['left',-.70],['right',.70]] as const){
    const lens=new THREE.Mesh(new THREE.BoxGeometry(.1,.1,.2));lens.name='taillamp_'+name+'__tail_lens';
    lens.position.set(-1.8,.65,z);body.add(lens);
  }
  system.night(true);system.visible(false);system.headlights(body);
  const running=u.uRearPower.value as number;
  assert.ok(running>0);
  assert.ok((u.uRearRight.value as THREE.Vector3).distanceTo(new THREE.Vector3(-2.3,8.65,12.8))<1e-6);
  emitter.emissiveIntensity=5;system.headlights(body);assert.ok((u.uRearPower.value as number)>running*3);
  emitter.dispose();
  assert.equal(u.uStreetOn.value,0);assert.equal(u.uBeamOn.value,1);
  assert.ok((u.uBeamLeft.value as THREE.Vector3).distanceTo(new THREE.Vector3(-3.75,8.6,9.3))<1e-6);
  assert.ok((u.uBeamForward.value as THREE.Vector3).distanceTo(new THREE.Vector3(0,0,-1))<1e-6);
  system.night(false);assert.equal(u.uBeamOn.value,0);
  system.night(true);system.headlights(new THREE.Group());assert.equal(u.uBeamOn.value,0);
  body.traverse(o=>{if(o instanceof THREE.Mesh)o.geometry.dispose();});
  system.dispose();material.dispose();
});

test('only a complete stack gets active lenses, aligned with its housing and the perpendicular approach',()=>{
  const parts:THREE.Mesh[]=[];
  function head(x:number,nx:number,nz:number,complete:boolean){
    const add=(name:string,px:number,py:number,pz:number,w:number,h:number,d:number)=>{
      const m=new THREE.Mesh(new THREE.BoxGeometry(w,h,d));m.name='street_objects___'+name;
      m.position.set(px,py,pz);parts.push(m);
    };
    add('charcoal',x,3,0,.3,1,.3);
    for(const [i,kind] of ['#7d2925','#92723a','#49836c'].entries()){
      if(i===2 && !complete)continue;
      add(kind,x+.19*nx,3.27-.27*i,.19*nz,.16,.17,.16);
    }
  }
  head(0,1,0,true);head(8,0,1,true);head(14,-1,0,false);
  const system=createStreetLights(parts,null,()=>0);
  const signal=system.group.getObjectByName('traffic-signal-active-lenses') as THREE.Mesh<THREE.InstancedBufferGeometry>;
  const phases=signal.geometry.getAttribute('signalPhase');
  assert.equal(signal.geometry.instanceCount,6);
  assert.equal(Math.round(Math.abs(phases.getY(3)-phases.getY(0))),32);
  assert.deepEqual([0,1,2,3,4,5].map(i=>phases.getX(i)),[0,1,2,0,1,2]);
  assert.equal(system.group.visible,true);
  system.visible(false);assert.equal(system.group.visible,false);
  system.dispose();for(const p of parts){p.geometry.dispose();(p.material as THREE.Material).dispose();}
});

test('uncertain window normals resolve from building occupancy and only low windows illuminate the outside pavement',()=>{
  const field={cell:.25,n:80,origin:-10,mask:new Uint8Array(80*80)};
  for(let j=0;j<80;j++)field.mask.fill(1,j*80,j*80+40);
  const low:GlazingEmitter={center:[0,13,0],normal:[-1,0],normalKnown:false,width:2,height:1.5,intensity:1,color:[1,.8,.6]};
  const high:GlazingEmitter={...low,center:[3,18,0]};
  const unknown=buildWindowSpill([low],()=>11);assert.equal(unknown.count,0);unknown.texture.dispose();
  const spill=buildWindowSpill([low,high],()=>11,field);
  assert.equal(spill.count,1);
  const {width,height,data}=spill.texture.image,b=spill.bounds;
  const index=(x:number,z:number)=>((Math.floor((z-b.y)*b.w*height)*width)+Math.floor((x-b.x)*b.z*width))*4;
  const val=(i:number)=>THREE.DataUtils.fromHalfFloat((data as Uint16Array)[i]);
  const lit=index(2,0);assert.ok(val(lit)>.10);assert.ok(val(lit+2)<val(lit));assert.ok(Math.abs(val(lit+3)/val(lit)-11)<.02);
  assert.equal(val(index(-2,0)),0);spill.texture.dispose();
});
