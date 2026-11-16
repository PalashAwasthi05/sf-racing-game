import * as THREE from 'three';

// Lens classes used by the two street-geometry exporters. A colour alone is
// insufficient: require three vertically aligned lenses and their backing box.
const ASPECTS = new Map([['#7d2925',0],['#92723a',1],['#49836c',2],['#b95246',0],['#c2a457',1],['#6a915f',2]]);
type Head = {centers: THREE.Vector3[]; normal: THREE.Vector2; radii: number[]};
type Junction = {center: THREE.Vector3; heads: Head[]};

export function signalHeads(meshes: THREE.Mesh[], parts: (mesh:THREE.Mesh)=>THREE.Box3[]) {
  const lenses: THREE.Box3[][]=[[],[],[]], housings: THREE.Box3[]=[];
  const sz=new THREE.Vector3(),c=new THREE.Vector3();
  for(const mesh of meshes) {
    const [prefix,kind]=mesh.name.split('___');
    if((mesh.userData.layer??prefix)!=='street_objects')continue;
    const aspect=ASPECTS.get(kind);
    if(aspect===undefined && kind!=='charcoal' && kind!=='dark')continue;
    for(const box of parts(mesh)) {
      box.getSize(sz);
      if(aspect!==undefined && sz.y>.08 && sz.y<.32 && Math.max(sz.x,sz.z)<.35)lenses[aspect].push(box);
      if(aspect===undefined && sz.y>=.70 && sz.y<=1.5 && Math.max(sz.x,sz.z)<=.8)housings.push(box);
    }
  }
  const junctions: Junction[]=[];
  for(const red of lenses[0]) {
    const top=red.getCenter(new THREE.Vector3());
    const amber=lenses[1].find(b=>{b.getCenter(c);return Math.hypot(c.x-top.x,c.z-top.z)<.04 && top.y-c.y>.15 && top.y-c.y<.48;});
    if(!amber)continue;
    const middle=amber.getCenter(new THREE.Vector3()), gap=top.y-middle.y;
    const green=lenses[2].find(b=>{b.getCenter(c);return Math.hypot(c.x-top.x,c.z-top.z)<.04 && Math.abs(middle.y-c.y-gap)<.035;});
    if(!green)continue;
    let best=Infinity, normal: THREE.Vector2|null=null;
    for(const box of housings) {
      box.getCenter(c);if(Math.abs(middle.y-c.y)>.19)continue;
      const dx=middle.x-c.x,dz=middle.z-c.z,d=Math.hypot(dx,dz);
      if(d<.10||d>.34||d>=best)continue;
      best=d;normal=new THREE.Vector2(dx/d,dz/d);
    }
    if(!normal)continue;
    const head={centers:[top,middle,green.getCenter(new THREE.Vector3())],normal,radii:[red,amber,green].map(b=>b.getSize(sz).y*.5)};
    let group=junctions.find(j=>Math.hypot(j.center.x-top.x,j.center.z-top.z)<33);
    if(!group){group={center:top.clone(),heads:[]};junctions.push(group);}
    group.center.lerp(top,1/(group.heads.length+1));group.heads.push(head);
  }
  return junctions;
}

export function signalLights(junctions:Junction[]) {
  const g=new THREE.InstancedBufferGeometry();
  g.setAttribute('position',new THREE.Float32BufferAttribute([-1,-1,0,1,-1,0,1,1,0,-1,1,0],3));
  g.setIndex([0,1,2,0,2,3]);
  const xyz:number[]=[],axis:number[]=[],phase:number[]=[];
  for(const junction of junctions) {
    const major=junction.heads[0].normal;
    const seed=Math.sin(junction.center.x*12.9898+junction.center.z*78.233)*43758.5453;
    for(const head of junction.heads) {
      const shift=(seed-Math.floor(seed))*64+(Math.abs(major.dot(head.normal))>=Math.SQRT1_2?0:32);
      for(const [i,c] of head.centers.entries()) {
        xyz.push(c.x+head.normal.x*.034,c.y,c.z+head.normal.y*.034);
        axis.push(head.normal.x,head.normal.y,head.radii[i]);phase.push(i,shift);
      }
    }
  }
  g.setAttribute('center',new THREE.InstancedBufferAttribute(new Float32Array(xyz),3));
  g.setAttribute('facing',new THREE.InstancedBufferAttribute(new Float32Array(axis),3));
  g.setAttribute('signalPhase',new THREE.InstancedBufferAttribute(new Float32Array(phase),2));
  g.instanceCount=xyz.length/3;
  const time={value:0}, start=performance.now();
  const material=new THREE.ShaderMaterial({
    transparent:true, depthWrite:false, side:THREE.DoubleSide, forceSinglePass:true,
    blending:THREE.CustomBlending,blendSrc:THREE.OneFactor,blendDst:THREE.OneFactor,
    blendSrcAlpha:THREE.ZeroFactor,blendDstAlpha:THREE.OneFactor,
    uniforms:{seconds:time},vertexShader:`
      uniform float seconds;
      attribute vec3 center,facing; attribute vec2 signalPhase;
      varying vec2 lensUV; varying vec3 lensColor; varying float facingEye;
      void main(){
        float p=mod(seconds+signalPhase.y,64.);
        float litAspect=p<27.?2.:(p<30.?1.:0.);
        if(abs(signalPhase.x-litAspect)>.1){gl_Position=vec4(2.,2.,2.,1.);return;}
        lensColor=litAspect<.5?vec3(1.,.016,.004):litAspect<1.5?vec3(1.,.43,.005):vec3(.006,1.,.23);
        vec3 n=vec3(facing.x,0.,facing.y);
        facingEye=smoothstep(.02,.32,dot(normalize(cameraPosition-center),n));
        lensUV=position.xy*1.38;
        vec3 world=center+(lensUV.x*vec3(-n.z,0.,n.x)+vec3(0.,lensUV.y,0.))*facing.z;
        gl_Position=projectionMatrix*viewMatrix*vec4(world,1.);
      }`,fragmentShader:`
      varying vec2 lensUV; varying vec3 lensColor; varying float facingEye;
      void main(){
        float r=length(lensUV);
        float edge=max(.045,fwidth(r)*.65);
        float disk=(1.-smoothstep(.91-edge,.91+edge,r))*(2.7+.8*max(0.,1.-r*r));
        gl_FragColor=vec4(lensColor*(disk+.07*exp(-r*r*2.6))*facingEye,0.);
      }`
  });
  const mesh=new THREE.Mesh(g,material);
  mesh.name='traffic-signal-active-lenses';mesh.frustumCulled=false;mesh.renderOrder=2;
  mesh.onBeforeRender=()=>{time.value=((performance.now()-start)/1000)%64;};
  return mesh;
}
