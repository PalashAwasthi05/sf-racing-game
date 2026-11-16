import * as THREE from 'three';
import type {GLTFParser} from 'three/addons/loaders/GLTFLoader.js';

// Only weak ownership metadata is global. The actual image table belongs to one
// viewer and contains entries only while a parser/cell has a lease on them.
const pooledSources=new WeakSet<THREE.Source>();
export const markPooledImageSource=(source:THREE.Source)=>{pooledSources.add(source);};
export const isPooledImageSource=(source:THREE.Source)=>pooledSources.has(source);
type Entry={key:string;references:number;source?:THREE.Source;pending:Promise<THREE.Source>;state:'queued'|'running'|'ready'|'failed';start?:()=>void;cancel?:()=>void};
export function createSceneImagePool(options:{anisotropy?:number;maxConcurrentDecodes?:number}={}){
 const entries=new Map<string,Entry>(),queue:Entry[]=[];
 const maxConcurrent=Math.max(1,Math.floor(options.maxConcurrentDecodes??3));let active=0;
 const aborted=()=>new DOMException('Scene image lease released','AbortError');
 function pump(){
  while(active<maxConcurrent&&queue.length){
   const entry=queue.shift()!;
   if(!entry.references){entry.cancel?.();continue;}
   entry.state='running';active++;const start=entry.start;entry.start=undefined;start!();
  }
 }
 function release(entry:Entry){
  if(--entry.references)return;
  if(entries.get(entry.key)===entry)entries.delete(entry.key);
  if(entry.source){(entry.source.data as ImageBitmap|null)?.close();entry.source.data=null;}
  else if(entry.state==='queued'){
   const index=queue.indexOf(entry);if(index>=0)queue.splice(index,1);
   entry.cancel?.();entry.cancel=undefined;entry.start=undefined;
  }
 }
 function retain(key:string,decode:()=>Promise<ImageBitmap>){
  let entry=entries.get(key);
  if(!entry){
   const created={key,references:0,state:'queued'} as Entry;
   let resolve!:(source:THREE.Source)=>void,reject!:(reason:unknown)=>void;
   created.pending=new Promise<THREE.Source>((done,fail)=>{resolve=done;reject=fail;});
   created.cancel=()=>{created.state='failed';reject(aborted());};
   created.start=()=>{
    created.cancel=undefined;
    void Promise.resolve().then(decode).then(image=>{
     const source=new THREE.Source(image);pooledSources.add(source);created.source=source;created.state='ready';
     if(!created.references){image.close();source.data=null;}
     resolve(source);
    },error=>{created.state='failed';if(entries.get(key)===created)entries.delete(key);reject(error);}).finally(()=>{active--;pump();});
   };
   entries.set(key,created);queue.push(created);queueMicrotask(pump);entry=created;
  }
  entry.references++;return entry;
 }
 return {anisotropy:options.anisotropy??1,retain,release,
  stats(){return {entries:entries.size,references:[...entries.values()].reduce((sum,entry)=>sum+entry.references,0)};},
  decodeStats(){return {active,queued:queue.length,maxConcurrent};},
 };
}
export type SceneImagePool=ReturnType<typeof createSceneImagePool>;
class ImageLease{
 private entries=new Map<string,Entry>();
 private released=false;
 private pool:SceneImagePool;
 constructor(pool:SceneImagePool){this.pool=pool;}
 async source(key:string,decode:()=>Promise<ImageBitmap>){
  if(this.released)throw new DOMException('Scene image lease released','AbortError');
  let entry=this.entries.get(key);
  if(!entry){entry=this.pool.retain(key,decode);this.entries.set(key,entry);}
  const source=await entry.pending;
  if(this.released)throw new DOMException('Scene image lease released','AbortError');
  return source;
 }
 prune(active?:ReadonlySet<THREE.Source>){
  for(const [key,entry] of this.entries)if(!active||!entry.source||!active.has(entry.source)){
   this.pool.release(entry);this.entries.delete(key);
  }
  if(!active)this.released=true;
 }
}
const parserLeases=new WeakMap<GLTFParser,ImageLease>();
const releasedParsers=new WeakSet<GLTFParser>();
type ParserResource={source:()=>THREE.Source|undefined;release:()=>void};
const parserResources=new WeakMap<GLTFParser,Map<object,ParserResource>>();
/** Additional exact image representations (DDS mip chains) share the same
 * parser lifecycle as portable images, including partial-load failure. */
export function registerParserImageResource(parser:GLTFParser,key:object,resource:ParserResource){
 if(releasedParsers.has(parser))throw new DOMException('Scene image lease released','AbortError');
 let resources=parserResources.get(parser);if(!resources){resources=new Map();parserResources.set(parser,resources);}
 if(resources.has(key))return false;resources.set(key,resource);return true;
}
function pruneParserResources(parser:GLTFParser,active?:ReadonlySet<THREE.Source>){
 const resources=parserResources.get(parser);if(!resources)return;
 for(const [key,resource] of resources)if(!active||!resource.source()||!active.has(resource.source()!)){
  resources.delete(key);resource.release();
 }
 if(!resources.size)parserResources.delete(parser);
}
/** Called after tint/reveal substitutions, before the cell can be installed. */
export function retainParserImageSources(parser:GLTFParser,materials:Iterable<THREE.Material>){
 const active=new Set<THREE.Source>();
 for(const material of materials)for(const value of Object.values(material))if(value instanceof THREE.Texture)active.add(value.source);
 parserLeases.get(parser)?.prune(active);pruneParserResources(parser,active);
}
export function releaseParserImageSources(parser:GLTFParser){releasedParsers.add(parser);parserLeases.get(parser)?.prune();parserLeases.delete(parser);pruneParserResources(parser);}

const filters:Record<number,THREE.TextureFilter>={9728:THREE.NearestFilter,9729:THREE.LinearFilter,9984:THREE.NearestMipmapNearestFilter,9985:THREE.LinearMipmapNearestFilter,9986:THREE.NearestMipmapLinearFilter,9987:THREE.LinearMipmapLinearFilter};
const wraps:Record<number,THREE.Wrapping>={33071:THREE.ClampToEdgeWrapping,33648:THREE.MirroredRepeatWrapping,10497:THREE.RepeatWrapping};
type ImageDef={bufferView?:number;mimeType?:string;extras?:{viewerImageSha256?:string}};
const digest=async(bytes:ArrayBuffer)=>Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256',bytes)),value=>value.toString(16).padStart(2,'0')).join('');

/** Preserve GLTFLoader's ImageBitmapLoader decode options exactly. Browsers
 * where Three selects its HTML-image fallback retain that original loader.
 * Different samplers/UV/color-space roles keep their own Texture wrappers;
 * Three reuses GPU storage for a shared Source with matching upload settings. */
export function pooledImagePlugin(parser:GLTFParser,pool:SceneImagePool){
 const lease=new ImageLease(pool);parserLeases.set(parser,lease);
 const json=parser.json as {images?:ImageDef[];textures?:{source?:number;sampler?:number;name?:string}[];samplers?:{magFilter?:number;minFilter?:number;wrapS?:number;wrapT?:number}[]};
 const pending=new Map<string,Promise<THREE.Texture>>();
 return {name:'viewer_shared_portable_images',afterRoot(){pending.clear();return null;},
  loadTexture(index:number){
   const definition=json.textures?.[index],image=definition?.source===undefined?undefined:json.images?.[definition.source];
   const bitmapLoader=(parser as unknown as {textureLoader?:{isImageBitmapLoader?:boolean}}).textureLoader?.isImageBitmapLoader;
   if(!definition||image?.bufferView===undefined||!/^image\/(?:png|jpeg|webp)$/.test(image.mimeType??'')||!bitmapLoader)return null;
   const localKey=definition.source+':'+definition.sampler;const cached=pending.get(localKey);if(cached)return cached;
   const result=(async()=>{
    let bytes:ArrayBuffer|undefined;
    const hash=image.extras?.viewerImageSha256;
    const identity=hash&&/^[a-f0-9]{64}$/i.test(hash)?hash.toLowerCase():await digest(bytes=await parser.getDependency('bufferView',image.bufferView!) as ArrayBuffer);
    // MIME and decode semantics are part of identity, not only encoded bytes.
    const source=await lease.source(`${identity}:${image.mimeType}:premultiply-none:colorspace-none:orientation-default`,async()=>{
     bytes??=await parser.getDependency('bufferView',image.bufferView!) as ArrayBuffer;
     return createImageBitmap(new Blob([bytes],{type:image.mimeType}),{premultiplyAlpha:'none',colorSpaceConversion:'none'});
    });
    const texture=new THREE.Texture();
    // Increment the wrapper's upload version before attaching the shared Source:
    // adding a neighbor must not mark an unchanged image dirty on existing cells.
    texture.needsUpdate=true;texture.source=source;
    const sampler=json.samplers?.[definition.sampler??-1]??{};
    texture.flipY=false;texture.anisotropy=pool.anisotropy;
    texture.magFilter=(filters[sampler.magFilter??9729]??THREE.LinearFilter) as THREE.MagnificationTextureFilter;
    texture.minFilter=filters[sampler.minFilter??9987]??THREE.LinearMipmapLinearFilter;
    texture.wrapS=wraps[sampler.wrapS??10497]??THREE.RepeatWrapping;texture.wrapT=wraps[sampler.wrapT??10497]??THREE.RepeatWrapping;
    texture.generateMipmaps=texture.minFilter!==THREE.NearestFilter&&texture.minFilter!==THREE.LinearFilter;
    texture.name=definition.name??'';texture.userData={...image.extras,mimeType:image.mimeType};
    parser.associations.set(texture,{textures:index});return texture;
   })();
   pending.set(localKey,result);return result;
  },
 };
}
