import assert from 'node:assert/strict';
import test from 'node:test';
import * as THREE from 'three';
import {paneAttributes, type GlazingEmitter} from './glazing.ts';

function windowMesh(x: number, y: number, z: number, angle: number) {
  const mesh = new THREE.Mesh(new THREE.PlaneGeometry(1.2, 2.1), new THREE.MeshStandardMaterial());
  mesh.position.set(x, y, z); mesh.rotation.y = angle;
  return mesh;
}

test('room coordinates follow a window rather than world storey boundaries', () => {
  const a = windowMesh(0, 3.5, 0, 0), b = windowMesh(211.73, 18.23, -54.6, 0);
  paneAttributes([a, b]);
  for (const mesh of [a,b]) {
    const p=mesh.geometry.getAttribute('glPane');
    for(let i=0;i<p.count;i++) {
      assert.ok(Math.abs(p.getZ(i)-1.2)<1e-4);
      assert.ok(Math.abs(p.getW(i)-2.1)<1e-4);
      assert.ok(p.getX(i)===0 || p.getX(i)===1);
      assert.ok(p.getY(i)===0 || p.getY(i)===1);
      // The entire measured opening stays strictly inside the interior room.
      const localY=p.getY(i)*p.getW(i);
      assert.ok(localY>-1 && localY<p.getW(i)+.6);
    }
  }
  assert.deepEqual(Array.from(a.geometry.getAttribute('glPane').array).map(x=>Math.round(x*1000)),
    Array.from(b.geometry.getAttribute('glPane').array).map(x=>Math.round(x*1000)));
});

test('rotated window room tangent agrees with increasing pane coordinates', () => {
  for(const angle of [Math.PI/3,-Math.PI/4,Math.PI*.8]) {
    const mesh=windowMesh(25,11,-31,angle); paneAttributes([mesh]);
    const p=mesh.geometry.getAttribute('glPane'), axis=mesh.geometry.getAttribute('glAxis');
    const pos=mesh.geometry.getAttribute('position');
    const left=new THREE.Vector3().fromBufferAttribute(pos,0).applyMatrix4(mesh.matrixWorld);
    const right=new THREE.Vector3().fromBufferAttribute(pos,1).applyMatrix4(mesh.matrixWorld);
    const projected=(right.x-left.x)*axis.getX(0)+(right.z-left.z)*axis.getY(0);
    assert.ok(Math.abs(projected-(p.getX(1)-p.getX(0))*p.getZ(0))<1e-4);
    assert.ok(Math.abs(Math.hypot(axis.getX(0),axis.getY(0))-1)<1e-6);
  }
});

test('panes separated by a mullion share one room without joining separate storeys',()=>{
  const a=new THREE.PlaneGeometry(.55,2.1), b=new THREE.PlaneGeometry(.55,2.1);
  a.translate(-.325,0,0); b.translate(.325,0,0);
  const joined=new THREE.BufferGeometry();
  const coords=[...a.toNonIndexed().attributes.position.array,...b.toNonIndexed().attributes.position.array];
  joined.setAttribute('position',new THREE.Float32BufferAttribute(coords,3));
  const mesh=new THREE.Mesh(joined); const windows=paneAttributes([mesh]);
  assert.equal(windows.length,1);
  assert.ok(Math.abs(windows[0].u1-windows[0].u0-1.2)<1e-4);
  assert.equal(new Set(joined.getAttribute('glId').array).size,1);
});

test('night spill follows the shader occupancy, tint and direction of each source window',()=>{
  const windows = Array.from({length:40}, (_, i) => windowMesh(i*2.5, 1.8, -4, Math.PI*.7));
  const emitters: GlazingEmitter[] = [];
  const classes = new Map<THREE.Material,string>();
  for (const mesh of windows) classes.set(mesh.material, 'glass_dark');
  paneAttributes(windows,emitters,classes);
  assert.ok(emitters.length>5 && emitters.length<35);
  for (const mesh of windows) {
    const code = mesh.geometry.getAttribute('glId').getX(0);
    const source = emitters.find(e => Math.abs(e.center[0]-mesh.position.x)<.001);
    assert.equal(!!source, (code & 255)<123);
    if (!source) continue;
    assert.ok(source.normalKnown);
    assert.ok(Math.abs(source.normal[0]-Math.sin(mesh.rotation.y))<1e-4);
    assert.ok(Math.abs(source.normal[1]-Math.cos(mesh.rotation.y))<1e-4);
    assert.ok(Math.abs(source.intensity-.42*.92*(.47+.63*Math.floor(code/65536)/255))<1e-5);
    assert.ok(Math.abs(source.color[1]-(.49+.34*((code>>>8)&255)/255))<1e-5);
  }
});

test('double-sided and textured windows do not claim outward irradiance',()=>{
  const transparent = windowMesh(0,0,0,0); // The origin carries lighting code zero.
  transparent.material.side = THREE.DoubleSide;
  const emitters: GlazingEmitter[] = [];
  paneAttributes([transparent],emitters);
  assert.equal(emitters.length,1);
  assert.equal(emitters[0].normalKnown,false);
  const painted = windowMesh(0,0,0,0);
  painted.material.map = new THREE.Texture();
  paneAttributes([painted],emitters);
  assert.equal(emitters.length,1);
});

test('live pane slicing preserves exact room coordinates, windows and emitting apertures',async()=>{
 const {paneAttributesAsync}=await import('./glazing.ts');
 const template=new THREE.BoxGeometry(1,2,.05),p=template.attributes.position,idx=template.index!,count=1800;
 const position=new Float32Array(p.array.length*count),index=new Uint32Array(idx.count*count);
 for(let k=0;k<count;k++){
  for(let i=0;i<p.count;i++){position[(k*p.count+i)*3]=p.getX(i)+(k%60)*3;position[(k*p.count+i)*3+1]=p.getY(i)+Math.floor(k/60)*3;position[(k*p.count+i)*3+2]=p.getZ(i);}
  for(let i=0;i<idx.count;i++)index[k*idx.count+i]=idx.getX(i)+k*p.count;
 }
 const geometry=new THREE.BufferGeometry().setAttribute('position',new THREE.BufferAttribute(position,3));geometry.setIndex(new THREE.BufferAttribute(index,1));
 const material=new THREE.MeshStandardMaterial();material.name='glass_dark';
 const ordinary=new THREE.Mesh(geometry,material),live=new THREE.Mesh(geometry.clone(),material);
 const before:import('./glazing.ts').GlazingEmitter[]=[],after:import('./glazing.ts').GlazingEmitter[]=[];
 const expected=paneAttributes([ordinary],before);let yields=0;
 const actual=await paneAttributesAsync([live],after,undefined,async()=>{yields++;});
 assert.ok(yields>0);assert.deepEqual(actual,expected);assert.deepEqual(after,before);
 for(const name of ['glPane','glId','glAxis'])assert.deepEqual(live.geometry.attributes[name].array,ordinary.geometry.attributes[name].array);
});
