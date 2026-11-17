// Runs in the inspection page after X.load('boxster', ...): builds the game's car layout and exports a GLB.
import * as THREE from 'three';
import {GLTFExporter} from 'three/addons/exporters/GLTFExporter.js';
import {FBXLoader} from 'three/addons/loaders/FBXLoader.js';
import {mergeVertices} from 'three/addons/utils/BufferGeometryUtils.js';
const TEX='../dl/boxster/Porsche_718BoxsterS_2017_';
const load=u=>new Promise((res,rej)=>new THREE.TextureLoader().load(u,t=>{t.flipY=true;res(t);},undefined,rej));

// Keeps the triangles of the connected pieces (welded by position) for which keep(bbox, tris) is true.
function filterPieces(geo,keep){const g=geo.index?geo.toNonIndexed():geo;const P=g.attributes.position,n=P.count;const ids=new Int32Array(n),key=new Map();let m=0;
 for(let i=0;i<n;i++){const k=Math.round(P.getX(i)*1e4)+'_'+Math.round(P.getY(i)*1e4)+'_'+Math.round(P.getZ(i)*1e4);let id=key.get(k);if(id===undefined){id=m++;key.set(k,id);}ids[i]=id;}
 const par=new Int32Array(m).map((_,i)=>i);const find=a=>{while(par[a]!==a){par[a]=par[par[a]];a=par[a];}return a;};
 for(let i=0;i<n;i+=3){const a=find(ids[i]);par[find(ids[i+1])]=a;par[find(ids[i+2])]=a;}
 const comp=new Map();for(let i=0;i<n;i+=3){const r=find(ids[i]);let c=comp.get(r);if(!c){c={tris:0,min:new THREE.Vector3(9,9,9),max:new THREE.Vector3(-9,-9,-9)};comp.set(r,c);}c.tris++;for(let k=i;k<i+3;k++){const v=new THREE.Vector3().fromBufferAttribute(P,k);c.min.min(v);c.max.max(v);}}
 const ok=new Map();for(const [r,c] of comp)ok.set(r,keep(c));
 const sel=[];for(let i=0;i<n;i+=3)if(ok.get(find(ids[i])))sel.push(i,i+1,i+2);
 const out=new THREE.BufferGeometry();for(const [name,a] of Object.entries(g.attributes)){const arr=new a.array.constructor(sel.length*a.itemSize);for(let j=0;j<sel.length;j++)for(let d=0;d<a.itemSize;d++)arr[j*a.itemSize+d]=a.array[sel[j]*a.itemSize+d];out.setAttribute(name,new THREE.BufferAttribute(arr,a.itemSize));}
 return out;}

// ---- donor parts from the Cayman GTS (same body, panels named) --------------------------------------------------
const CAY='../dl/cayman/';
async function loadDonor(CAY='../dl/cayman/',file='FINAL_MODEL_GTS.fbx'){const mgr=new THREE.LoadingManager();mgr.setURLModifier(u=>{const n=u.split(/[\\/]/).pop();return /\.(png|jpg|jpeg|tga|dds)$/i.test(n)?CAY+n:u;});
 const o=await new FBXLoader(mgr).loadAsync(CAY+file);o.rotation.y=Math.PI/2;o.position.set(.0265,-.03,0);o.updateMatrixWorld(true);
 const parts=[];o.traverse(m=>{if(!m.isMesh)return;let top=m;while(top.parent&&top.parent!==o)top=top.parent;const g=m.geometry.clone();g.applyMatrix4(m.matrixWorld);
  const mats=[].concat(m.material);if(mats.length===1||!g.groups.length)parts.push({top:top.name,mat:mats[0],geo:g.index?g.toNonIndexed():g});
  else{const ni=g.index?g.toNonIndexed():g;for(const gr of g.groups){const sub=new THREE.BufferGeometry();for(const [k,a] of Object.entries(ni.attributes))sub.setAttribute(k,new THREE.BufferAttribute(a.array.slice(gr.start*a.itemSize,(gr.start+gr.count)*a.itemSize),a.itemSize));parts.push({top:top.name,mat:mats[gr.materialIndex],geo:sub});}}});
 return parts;}
// nearest-panel classifier: samples of the donor's panels in a hash grid, each tagged replace (1) or keep (0)
function classifier(parts,replace,keep){const cell=.05,grid=new Map();const put=(x,y,z,c)=>{const k=Math.floor(x/cell)+','+Math.floor(y/cell)+','+Math.floor(z/cell);let a=grid.get(k);if(!a){a=[];grid.set(k,a);}a.push(x,y,z,c);};
 const a=new THREE.Vector3(),b=new THREE.Vector3(),c=new THREE.Vector3();
 for(const p of parts){const cat=replace.test(p.top)?1:keep.test(p.top)?0:-1;if(cat<0)continue;const P=p.geo.attributes.position;
  for(let i=0;i<P.count;i+=3){a.fromBufferAttribute(P,i);b.fromBufferAttribute(P,i+1);c.fromBufferAttribute(P,i+2);const n=Math.min(14,Math.max(1,Math.ceil(Math.max(a.distanceTo(b),b.distanceTo(c),c.distanceTo(a))/.018)));
   for(let u=0;u<=n;u++)for(let v=0;v<=n-u;v++){const w=n-u-v;put((a.x*u+b.x*v+c.x*w)/n,(a.y*u+b.y*v+c.y*w)/n,(a.z*u+b.z*v+c.z*w)/n,cat);}}}
 return (x,y,z)=>{let best=1e9,cat=-1;const cx=Math.floor(x/cell),cy=Math.floor(y/cell),cz=Math.floor(z/cell);
  for(let i=-2;i<=2;i++)for(let j=-2;j<=2;j++)for(let k=-2;k<=2;k++){const arr=grid.get((cx+i)+','+(cy+j)+','+(cz+k));if(!arr)continue;for(let q=0;q<arr.length;q+=4){const d=(arr[q]-x)**2+(arr[q+1]-y)**2+(arr[q+2]-z)**2;if(d<best){best=d;cat=arr[q+3];}}}
  return cat;};}
function dropTriangles(geo,drop){const g=geo.index?geo.toNonIndexed():geo;const P=g.attributes.position;const sel=[];let removed=0;
 for(let i=0;i<P.count;i+=3){const x=(P.getX(i)+P.getX(i+1)+P.getX(i+2))/3,y=(P.getY(i)+P.getY(i+1)+P.getY(i+2))/3,z=(P.getZ(i)+P.getZ(i+1)+P.getZ(i+2))/3;if(drop(x,y,z))removed++;else sel.push(i,i+1,i+2);}
 if(!removed)return {geo,removed};const out=new THREE.BufferGeometry();for(const [name,a] of Object.entries(g.attributes)){const arr=new a.array.constructor(sel.length*a.itemSize);for(let j=0;j<sel.length;j++)for(let d=0;d<a.itemSize;d++)arr[j*a.itemSize+d]=a.array[sel[j]*a.itemSize+d];out.setAttribute(name,new THREE.BufferAttribute(arr,a.itemSize));}
 return {geo:out,removed};}
// The rear model lettering is one small strip across the tail (texture rows v .79 to .91). Widens it about the centre
// line by ratio, following the tail's curve: x = a + b*y + c*z*z is fitted to the strip and c carries the new z.
function widenRearBadge(geo,ratio,report){const P=geo.attributes.position,U=geo.attributes.uv,idx=[];
 for(let i=0;i<P.count;i++){const v=U.getY(i);if(P.getX(i)<-2&&v>.78&&v<.915&&U.getX(i)>=0&&Math.abs(P.getZ(i))<.2&&P.getY(i)<.72)idx.push(i);}
 if(idx.length<4)return;
 // least squares for a, b, c
 const A=[[0,0,0],[0,0,0],[0,0,0]],B=[0,0,0];for(const i of idx){const r=[1,P.getY(i),P.getZ(i)**2],x=P.getX(i);for(let j=0;j<3;j++){B[j]+=r[j]*x;for(let k=0;k<3;k++)A[j][k]+=r[j]*r[k];}}
 const det=m=>m[0][0]*(m[1][1]*m[2][2]-m[1][2]*m[2][1])-m[0][1]*(m[1][0]*m[2][2]-m[1][2]*m[2][0])+m[0][2]*(m[1][0]*m[2][1]-m[1][1]*m[2][0]);
 const d=det(A),M=A.map(r=>r.slice());for(let j=0;j<3;j++)M[j][2]=B[j];const c=Math.abs(d)>1e-12?det(M)/d:0;
 for(const i of idx){const z=P.getZ(i),z2=z*ratio;P.setX(i,P.getX(i)+c*(z2*z2-z*z)-.0015);P.setZ(i,z2);}
 P.needsUpdate=true;report.push(['badge',idx.length,'curve',+c.toFixed(3)]);}
export async function convert(opts={}){
 const root=window.X.root;root.updateMatrixWorld(true);
 const lift=opts.lift??.026, shiftX=opts.shiftX??.0475;
 const lightE=await load(TEX+'Light_Emissive.png');lightE.colorSpace=THREE.SRGBColorSpace;
 const wheelN=await load(TEX+'Wheel1A_Normal.png');const interiorN=await load(TEX+'Interior_Normal.png');
 // gtsBadge: the sheet from make_gts_badge.py, whose rear lettering is opts.gtsBadge times as long as the original
 let badgeMap=null;if(opts.gtsBadge){badgeMap=await load('../dl/boxster/Badge_GTS.png');badgeMap.colorSpace=THREE.SRGBColorSpace;}
 const std=o=>new THREE.MeshStandardMaterial(o);const made=new Map();
 const srcMat=m=>{if(made.has(m.name))return made.get(m.name);let r;
  if(m.name==='coloured')r=std({map:m.map,roughness:.5,metalness:0});
  else if(m.name==='interior')r=std({map:m.map,normalMap:interiorN,roughness:.82,metalness:0});
  else if(m.name==='light')r=std({map:m.map,emissiveMap:lightE,emissive:'#ffffff',emissiveIntensity:opts.lightGlow??.9,roughness:.25,metalness:.2});
  else if(m.name==='wheel')r=std({map:m.map,normalMap:wheelN,color:opts.wheelColor??'#ffffff',roughness:.42,metalness:.55});
  else if(m.name==='badges')r=std({map:badgeMap??m.map,transparent:true,alphaTest:.4,roughness:.3,metalness:.6});
  else r=std({map:m.map,color:m.map?'#ffffff':m.color,roughness:.6,metalness:0});
  r.name=m.name;made.set(m.name,r);return r;};
 const out=new THREE.Group();const report=[];
 // one mesh per (object, material group), world transform baked in
 const pieces=[];root.traverse(o=>{if(!o.isMesh)return;const mats=Array.isArray(o.material)?o.material:[o.material];
  const base=o.geometry.clone();base.applyMatrix4(o.matrixWorld);
  if(mats.length===1||!base.groups.length){pieces.push({o,mat:mats[0],geo:base});return;}
  const idx=base.index?base.index.array:null;
  for(const g of base.groups){const sub=base.clone();sub.clearGroups();
   if(idx)sub.setIndex(Array.from(idx.slice(g.start,g.start+g.count)));
   else{for(const k of Object.keys(sub.attributes)){const a=base.attributes[k];sub.setAttribute(k,new THREE.BufferAttribute(a.array.slice(g.start*a.itemSize,(g.start+g.count)*a.itemSize),a.itemSize));}}
   pieces.push({o,mat:mats[g.materialIndex],geo:sub});}});
 const path=o=>{const d=[];let p=o;while(p&&p!==root){d.push(p.name);p=p.parent;}return d.reverse().join('/');};
 // wheel centres from the tyre (largest piece) of the two right-hand groups
 const groups={front:'group1',rear:'group4'};const centre={};
 for(const [axle,g] of Object.entries(groups)){let best=null;for(const p of pieces){if(!path(p.o).includes('/'+g+'/'))continue;p.geo.computeBoundingBox();const b=p.geo.boundingBox,s=(b.max.x-b.min.x)*(b.max.z-b.min.z);if(!best||s>best.s)best={s,b};}
  centre[axle]=new THREE.Vector3((best.b.min.x+best.b.max.x)/2,(best.b.min.y+best.b.max.y)/2,(best.b.min.z+best.b.max.z)/2);report.push([axle,centre[axle].toArray().map(v=>+v.toFixed(4)),+((best.b.max.x-best.b.min.x)/2).toFixed(4)]);}
 let donor=null,isFront=null;const REPLACE=/^pBumperF6|^pHeadLight|^pBumperChassisF6/,KEEP=/^pFenderF|^pHood6|^pFendersChassisF6|^pChassis6|^pSkirts6|^pDoor|^pMirror|^pInterior|^pGlass/;
 if(opts.gtsFront){donor=await loadDonor();const near=classifier(donor,REPLACE,KEEP);isFront=(x,y,z)=>x>1.25&&near(x,y,z)===1;}
 let isDiff=null;
 if(opts.gtsDiffuser){donor??=await loadDonor();const nearR=classifier(donor,/^pDiffuser6|^pExhausts6/,/^pBumperR6|^pBumperChassisR6|^pFendersR6|^pBoot6|^pTailLight|^pSpoiler6|^pChassis6|^pSkirts6/);isDiff=(x,y,z)=>x<-1.6&&y<.5&&nearR(x,y,z)===1;}
 let tailMat=null;
 let n=0;const add=(name,geo,mat,extra)=>{geo.clearGroups();for(const k of Object.keys(geo.attributes))if(!['position','normal','uv'].includes(k))geo.deleteAttribute(k);geo=mergeVertices(geo,1e-5);const m=new THREE.Mesh(geo,mat);m.name=name;if(extra)m.userData=extra;out.add(m);n++;};
 for(const p of pieces){const pp=path(p.o),mn=p.mat.name;
  const wheel=pp.includes('Combined1_Src4/');
  if(wheel){const axle=pp.includes('/group1/')?'front':pp.includes('/group4/')?'rear':null;if(!axle)continue;
   p.geo.translate(-centre[axle].x,-centre[axle].y,-centre[axle].z);
   if(mn==='calipers')add('hub_'+axle+'__caliper',p.geo,null);
   else if(mn==='b')add('hub_'+axle+'__black_satin',p.geo,null);
   else add('wheel_'+axle+'__src_'+n,p.geo,srcMat(p.mat));
   continue;}
  if(mn==='plate'||mn==='phong14')continue;
  if(pp.includes('WindowInside'))continue;
  if(opts.open){
   // roof down: no soft top, no rear or side glass (the windscreen stays)
   if(pp.includes('Coloured_Geo'))p.geo=filterPieces(p.geo,c=>!(c.max.y>1.2&&c.min.y>.8&&c.max.x<.3)&&!(opts.dropTrim&&c.min.y>.8&&c.max.y<1.02&&c.max.x<-.5&&c.min.x<-1.1));
   if(pp.includes('Window_Geo'))p.geo=filterPieces(p.geo,c=>c.max.x>.6);
  }
  if(opts.gtsPipes&&/Coloured_Geo|Base_Geo|Grille/.test(pp)){p.geo=filterPieces(p.geo,c=>!(c.max.x<-1.85&&c.max.y<.34&&Math.max(-c.min.z,c.max.z)<.15));if(!p.geo.attributes.position.count)continue;}
  if(isDiff){p.geo.computeBoundingBox();if(p.geo.boundingBox.min.x<-1.6&&!/Paint_Geo/.test(pp)){const r=dropTriangles(p.geo,isDiff);if(r.removed){report.push(['cutR',pp.slice(-30)+'/'+mn,r.removed]);p.geo=r.geo;}if(!p.geo.attributes.position.count)continue;}}
  if(isFront){p.geo.computeBoundingBox();if(p.geo.boundingBox.max.x>1.25){const r=dropTriangles(p.geo,isFront);if(r.removed){report.push(['cut',pp.slice(-30)+'/'+mn,r.removed]);p.geo=r.geo;}if(!p.geo.attributes.position.count)continue;}}
  if(opts.gtsBadge&&mn==='badges')widenRearBadge(p.geo,opts.gtsBadge,report);
  p.geo.translate(shiftX,lift,0);
  if(mn==='carpaint')add('body__paint',p.geo,null,{rearPlateMount:{position:[-2.168+shiftX-.004,.518+lift,0],normal:[-1,.06,0]}});
  else if(mn==='glass')add('glass__glass_'+n,p.geo,null);
  else if(mn==='red_glass')add('taillens__tail_lens_'+n,p.geo,null);
  else if(mn==='chassis')add('base__black_matte',p.geo,null);
  else if(mn==='standardSurface1')add('cabinzone__cabin_soft',p.geo,null);
  else if(mn==='light'){   // tail lamps get their own material so the game can brighten them under braking
   const rear=dropTriangles(p.geo,(x,y,z)=>x>=-1.5+shiftX),front=dropTriangles(p.geo,(x,y,z)=>x<-1.5+shiftX);
   if(front.geo.attributes.position.count)add('light__src_'+n,front.geo,srcMat(p.mat));
   if(rear.geo.attributes.position.count){if(!tailMat){tailMat=srcMat(p.mat).clone();tailMat.name='light_tail';tailMat.emissive.set('#ff1408');tailMat.emissiveIntensity=.8;}add('taillamp__src_'+n,rear.geo,tailMat);}}
  else add((pp.split('_S')[1]||'part').replace(/_Geo_lodA.*/,'').toLowerCase()+'__src_'+n,p.geo,srcMat(p.mat));}
 if(donor){const E=await load(CAY+'Car_Porsche_718CaymanGTS_2018_Light_SetA_E.png');E.colorSpace=THREE.SRGBColorSpace;let bucket=null;
  for(const p of donor){if(!REPLACE.test(p.top))continue;const mn=p.mat.name;if(/Badge/.test(mn))continue;p.geo.translate(shiftX,lift,0);const part=p.top.startsWith('pHeadLight')?'gtslamp':'gtsbumper';
   if(/M_CarPaint_Max/.test(mn))add(part+'__paint_'+n,p.geo,null);
   else if(/LightBucket/.test(mn)){bucket??=Object.assign(std({map:p.mat.map,emissiveMap:E,emissive:'#ffffff',emissiveIntensity:opts.lightGlow??.9,roughness:.3,metalness:.3}),{name:'gtslight'});add(part+'__src_'+n,p.geo,bucket);}
   else if(/LightGlass|Glass_WindowFront/.test(mn))add(part+'__lens_'+n,p.geo,null);
   else if(/WindowSurround|PlasticSmoothBlack/.test(mn))add(part+'__black_gloss_'+n,p.geo,null);
   else if(/Chassis/.test(mn))add(part+'__black_matte_'+n,p.geo,null);
   else add(part+'__black_satin_'+n,p.geo,null);}}
 if(opts.gtsDiffuser){const spy=await loadDonor('../dl/spyder/','FINAL_MODEL_2019.fbx');
  for(const p of spy){const mn=p.top+'|'+p.mat.name;if(/Wheel|Interior|Badge|Light|Glass|Carpaint|LicensePlate|Engine|Grille/i.test(mn))continue;p.geo.computeBoundingBox();if(p.geo.boundingBox.min.x>-1.6)continue;
   const r=dropTriangles(p.geo,(x,y,z)=>!isDiff(x,y,z));if(!r.geo.attributes.position.count)continue;report.push(['spyder',mn.slice(4,60),r.geo.attributes.position.count/3]);
   r.geo.translate(shiftX,lift,0);add('gtsrear__'+(/PaintSatinBlack/.test(mn)?'black_gloss':/Chassis/.test(mn)?'black_matte':'black_satin')+'_'+n,r.geo,null);}}
 if(opts.gtsPipes){const spy=await loadDonor('../dl/spyder/','FINAL_MODEL_2019.fbx');const lo=opts.pipeBox??{x:-1.84,y:.44,z:.75};
  for(const p of spy){if(!/PaintSatinBlack/.test(p.top+p.mat.name))continue;/* tips only: the Spyder's diffuser face is one piece with its bumper */p.geo.computeBoundingBox();const b=p.geo.boundingBox;
   if(b.min.x>lo.x||b.min.y>lo.y)continue;
   if(!(b.max.x<lo.x&&b.max.y<lo.y&&Math.max(-b.min.z,b.max.z)<lo.z)){p.geo=filterPieces(p.geo,c=>c.max.x<lo.x&&c.max.y<lo.y&&Math.max(-c.min.z,c.max.z)<lo.z&&c.min.y<.36);if(!p.geo.attributes.position.count)continue;p.geo.computeBoundingBox();}
   const b2=p.geo.boundingBox;report.push(['spyder',p.top.slice(0,40),p.mat.name.slice(0,34),p.geo.attributes.position.count/3,b2.min.toArray().map(v=>+v.toFixed(2)),b2.max.toArray().map(v=>+v.toFixed(2))]);
   p.geo.translate(shiftX,lift,0);const mn=p.top+p.mat.name;
   add('gtsrear__'+(/PaintSatinBlack|AluminumDark/.test(mn)?'black_gloss':/Chassis/.test(mn)?'black_matte':'black_satin')+'_'+n,p.geo,null);}}
 const placeholder=new THREE.MeshStandardMaterial({color:'#888'});out.traverse(o=>{if(o.isMesh&&!o.material)o.material=placeholder;});
 const glb=await new GLTFExporter().parseAsync(out,{binary:true});
 const u=new Uint8Array(glb);let s='';for(let i=0;i<u.length;i+=0x8000)s+=String.fromCharCode(...u.subarray(i,i+0x8000));
 return {report,meshes:n,bytes:u.length,b64:btoa(s)};
}
