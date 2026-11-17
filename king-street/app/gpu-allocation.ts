import type * as THREE from 'three';

/** Opt-in accounting of actual GL allocations. Keeps only currently live handles. */
export function gpuAllocationReport(gl:WebGL2RenderingContext){
 const buffers=new Map<WebGLBuffer,number>(),textures=new Map<WebGLTexture,Map<string,number>>(),renderbuffers=new Map<WebGLRenderbuffer,number>();
 const boundBuffers=new Map<number,WebGLBuffer|null>(),boundTextures=new Map<string,WebGLTexture|null>();
 let unit=0,boundRenderbuffer:WebGLRenderbuffer|null=null;
 const undo:(()=>void)[]=[];
 const hook=(name:string,after:(args:any[])=>void)=>{
  const api=gl as unknown as Record<string,(...args:any[])=>any>,original=api[name];
  api[name]=function(...args:any[]){const result=original.apply(gl,args);after(args);return result;};
  undo.push(()=>{api[name]=original;});
 };
 const root=(target:number)=>target>=gl.TEXTURE_CUBE_MAP_POSITIVE_X&&target<=gl.TEXTURE_CUBE_MAP_NEGATIVE_Z?gl.TEXTURE_CUBE_MAP:target;
 const tex=(target:number)=>boundTextures.get(unit+':'+root(target));
 const sizedBytes=new Map<number,number>([
  [gl.R8,1],[gl.RG8,2],[gl.RGB8,3],[gl.RGBA8,4],[gl.SRGB8_ALPHA8,4],
  [gl.R16F,2],[gl.RG16F,4],[gl.RGB16F,6],[gl.RGBA16F,8],
  [gl.R32F,4],[gl.RG32F,8],[gl.RGB32F,12],[gl.RGBA32F,16],
  [gl.R11F_G11F_B10F,4],[gl.RGB10_A2,4],[gl.DEPTH_COMPONENT16,2],
  [gl.DEPTH_COMPONENT24,4],[gl.DEPTH_COMPONENT32F,4],[gl.DEPTH24_STENCIL8,4],[gl.DEPTH32F_STENCIL8,8],
 ]);
 const bpp=(internal:number,format:number,type:number)=>{
  const known=sizedBytes.get(internal);if(known)return known;
  const components=format===gl.RED||format===gl.DEPTH_COMPONENT?1:format===gl.RG?2:format===gl.RGB?3:4;
  return components*(type===gl.FLOAT?4:type===gl.HALF_FLOAT||type===gl.UNSIGNED_SHORT?2:1);
 };
 function image(target:number,level:number,bytes:number){
  const t=tex(target);if(!t)return;
  const levels=textures.get(t)??new Map<string,number>();levels.set(target+':'+level,bytes);textures.set(t,levels);
 }
 hook('bindBuffer',([target,value])=>boundBuffers.set(target,value));
 hook('bufferData',([target,data,,,length])=>{const buffer=boundBuffers.get(target);if(buffer)buffers.set(buffer,typeof data==='number'?data:(length?length*(data.BYTES_PER_ELEMENT??1):data?.byteLength??0));});
 hook('deleteBuffer',([buffer])=>buffers.delete(buffer));
 hook('activeTexture',([value])=>{unit=value-gl.TEXTURE0;});
 hook('bindTexture',([target,value])=>boundTextures.set(unit+':'+target,value));
 hook('texImage2D',a=>{
  const [target,level,internal]=a;
  const width=a.length>=9?a[3]:a[5]?.width,height=a.length>=9?a[4]:a[5]?.height;
  image(target,level,(width??0)*(height??0)*bpp(internal,a.length>=9?a[6]:a[3],a.length>=9?a[7]:a[4]));
 });
 hook('compressedTexImage2D',a=>image(a[0],a[1],a[6]?.byteLength??a[6]??0));
 for(const name of ['texStorage2D','texStorage3D'])hook(name,([target,levels,internal,width,height,depth=1])=>{
  const count=target===gl.TEXTURE_CUBE_MAP?6:1;
  for(let level=0;level<levels;level++)image(target,level,count*Math.max(1,width>>level)*Math.max(1,height>>level)*depth*bpp(internal,internal,gl.UNSIGNED_BYTE));
 });
 hook('texImage3D',a=>image(a[0],a[1],a[3]*a[4]*a[5]*bpp(a[2],a[7],a[8])));
 hook('generateMipmap',([target])=>{
  const texture=tex(target),levels=texture&&textures.get(texture);if(!levels)return;
  if([...levels.keys()].some(key=>!key.endsWith(':0')))return;
  const base=[...levels].filter(([key])=>key.endsWith(':0')).reduce((n,[,v])=>n+v,0);
  levels.set('generated-mips',Math.ceil(base/3));
 });
 hook('deleteTexture',([texture])=>textures.delete(texture));
 hook('bindRenderbuffer',([,buffer])=>{boundRenderbuffer=buffer;});
 hook('renderbufferStorage',([,format,w,h])=>{if(boundRenderbuffer)renderbuffers.set(boundRenderbuffer,w*h*bpp(format,format,gl.UNSIGNED_BYTE));});
 hook('renderbufferStorageMultisample',([,samples,format,w,h])=>{if(boundRenderbuffer)renderbuffers.set(boundRenderbuffer,samples*w*h*bpp(format,format,gl.UNSIGNED_BYTE));});
 hook('deleteRenderbuffer',([buffer])=>renderbuffers.delete(buffer));
 const sum=(values:Iterable<number>)=>[...values].reduce((n,v)=>n+v,0);
 return {
  snapshot(renderer:THREE.WebGLRenderer,scene:THREE.Scene){
   const names=new Map<WebGLTexture,Set<string>>();
   function nameTexture(texture:any,label:string){
    if(!texture?.isTexture)return;
    const handle=(renderer.properties.get(texture) as {__webglTexture?:WebGLTexture}).__webglTexture;
    if(!handle)return;const labels=names.get(handle)??new Set<string>();labels.add(label);names.set(handle,labels);
   }
   scene.traverse(object=>{
    const material=(object as THREE.Mesh).material;
    for(const m of material?(Array.isArray(material)?material:[material]):[]){
     for(const [key,value] of Object.entries(m))nameTexture(value,m.name+':'+key);
     const compiled=renderer.properties.get(m) as {uniforms?:Record<string,{value:unknown}>};
     for(const [key,u] of Object.entries(compiled.uniforms??(m as THREE.ShaderMaterial).uniforms??{}))nameTexture(u.value,m.name+':'+key);
    }
   });
   const rows=[...textures].map(([handle,levels])=>({bytes:sum(levels.values()),labels:[...names.get(handle)??[]].slice(0,5)})).sort((a,b)=>b.bytes-a.bytes);
   return {bufferBytes:sum(buffers.values()),textureBytes:sum(rows.map(t=>t.bytes)),renderbufferBytes:sum(renderbuffers.values()),
    liveBuffers:buffers.size,liveTextures:textures.size,liveRenderbuffers:renderbuffers.size,largestTextures:rows.slice(0,18)};
  },
  dispose(){for(const restore of undo)restore();buffers.clear();textures.clear();renderbuffers.clear();}
 };
}
