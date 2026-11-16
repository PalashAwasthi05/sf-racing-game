import * as THREE from 'three';
import type {GLTFParser} from 'three/addons/loaders/GLTFLoader.js';
import {DDSLoader} from 'three/addons/loaders/DDSLoader.js';
import {markPooledImageSource,registerParserImageResource,type SceneImagePool} from './scene-image-pool.ts';

/** Both extensions are needed: color maps use sRGB S3TC while normals and
 * roughness maps use linear S3TC. Other devices keep the portable package. */
export function supportsStartupS3tc(renderer: Pick<THREE.WebGLRenderer, 'extensions'>) {
  return renderer.extensions.has('WEBGL_compressed_texture_s3tc') &&
    renderer.extensions.has('WEBGL_compressed_texture_s3tc_srgb');
}

// Sources survive sampler/UV clones without JSON-copying nonserializable data.
// Keep only encoded originals until merge asks for an exact browser decode.
const cpuImages = new WeakMap<THREE.Source, () => Promise<ImageBitmap>>();
// Exact DDS Sources may be resident in multiple parsers. Encoded tint images
// have a separate lease per parser; finishing one merge cannot remove another
// cell's still-needed decoder. Only weak registries are shared globally.
const pooledCpuImages=new WeakMap<THREE.Source,Map<GLTFParser,()=>Promise<ImageBitmap>>>();
const parserCpuImages=new WeakMap<GLTFParser,Set<THREE.Source>>();
function releaseParserCpuImage(parser:GLTFParser,source:THREE.Source){
 const decoders=pooledCpuImages.get(source);decoders?.delete(parser);if(!decoders?.size)pooledCpuImages.delete(source);
 const sources=parserCpuImages.get(parser);sources?.delete(source);if(!sources?.size)parserCpuImages.delete(parser);
}
function addParserCpuImage(parser:GLTFParser,source:THREE.Source,decode:()=>Promise<ImageBitmap>){
 let decoders=pooledCpuImages.get(source);if(!decoders){decoders=new Map();pooledCpuImages.set(source,decoders);}decoders.set(parser,decode);
 let sources=parserCpuImages.get(parser);if(!sources){sources=new Set();parserCpuImages.set(parser,sources);}sources.add(source);
}
type DdsEntry={pending:Promise<THREE.CompressedTexture>;base?:THREE.CompressedTexture;owners:Set<GLTFParser>};
const ddsPools=new WeakMap<SceneImagePool,Map<string,DdsEntry>>();
export function compressedImagePoolStats(pool:SceneImagePool){
 const entries=ddsPools.get(pool);return {entries:entries?.size??0,references:entries?[...entries.values()].reduce((sum,entry)=>sum+entry.owners.size,0):0};
}
async function sharedDdsTexture(parser:GLTFParser,pool:SceneImagePool,key:string,load:()=>THREE.CompressedTexture){
 let entries=ddsPools.get(pool);if(!entries){entries=new Map();ddsPools.set(pool,entries);}
 let entry=entries.get(key);
 if(!entry){
  const created={owners:new Set<GLTFParser>()} as DdsEntry;
  created.pending=Promise.resolve().then(load).then(base=>{created.base=base;markPooledImageSource(base.source);
   if(!created.owners.size){base.mipmaps.length=0;base.source.data=null;}return base;
  },error=>{if(entries!.get(key)===created)entries!.delete(key);throw error;});
  entries.set(key,created);entry=created;
 }
 const acquired=entry;
 try{
  if(registerParserImageResource(parser,entry,{source:()=>acquired.base?.source,release:()=>{
   if(acquired.base)releaseParserCpuImage(parser,acquired.base.source);
   acquired.owners.delete(parser);if(acquired.owners.size)return;
   if(entries!.get(key)===acquired)entries!.delete(key);
   if(acquired.base){acquired.base.mipmaps.length=0;acquired.base.source.data=null;}
  }}))entry.owners.add(parser);
 }catch(error){
  if(!entry.owners.size&&entries.get(key)===entry)entries.delete(key);
  // load() starts in a microtask; consume its outcome after failed registration.
  void entry.pending.catch(()=>{});throw error;
 }
 const base=await entry.pending;
 if(!entry.owners.has(parser))throw new DOMException('Scene image lease released','AbortError');
 return base;
}
const ddsDigest=async(bytes:ArrayBuffer)=>Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256',bytes)),value=>value.toString(16).padStart(2,'0')).join('');
function ddsMetadata(bytes:ArrayBuffer){
 if(bytes.byteLength<128)throw new Error('Invalid startup DDS image');const header=new DataView(bytes);
 // Full-image digest also covers every mip's payload/offset; these explicit
 // header fields prevent accidental sharing across format/layout roles.
 return [header.getUint32(84,true),header.getUint32(16,true),header.getUint32(12,true),header.getUint32(28,true),bytes.byteLength].join(':');
}
export async function readTextureCpuImage<T>(texture: THREE.Texture, read: (image: ImageBitmap) => T): Promise<T | null> {
  const decode = pooledCpuImages.get(texture.source)?.values().next().value ?? cpuImages.get(texture.source);
  if (!decode) return null;
  const image = await decode();
  try { return read(image); }
  finally { image.close(); }
}

type ImageDef = {bufferView: number; mimeType?: string; extras?: {viewerCpuBufferView?: number; viewerCpuMimeType?: string; viewerImageSha256?:string; viewerCpuImageSha256?:string}};
const filters: Record<number, THREE.TextureFilter> = {9728: THREE.NearestFilter, 9729: THREE.LinearFilter,
  9984: THREE.NearestMipmapNearestFilter, 9985: THREE.LinearMipmapNearestFilter,
  9986: THREE.NearestMipmapLinearFilter, 9987: THREE.LinearMipmapLinearFilter};
const wraps: Record<number, THREE.Wrapping> = {33071: THREE.ClampToEdgeWrapping, 33648: THREE.MirroredRepeatWrapping, 10497: THREE.RepeatWrapping};

type TextureRef = {index: number};
type CompanionDocument = {
  textures?: {source: number}[];
  materials?: {extensions?: object; normalTexture?: TextureRef; occlusionTexture?: TextureRef;
    pbrMetallicRoughness?: {baseColorTexture?: TextureRef; metallicRoughnessTexture?: TextureRef}}[];
};

/** Merge samples only these five core material maps. Emissive-only images never
 * need a decoded companion. Unknown material extensions keep the general path. */
export function mergeSampleImageSources(json: CompanionDocument): Set<number> | undefined {
  if (!json.materials || json.materials.some(material => material.extensions)) return undefined;
  const sources = new Set<number>();
  for (const material of json.materials) {
    const pbr = material.pbrMetallicRoughness;
    // Materials without a colour map are excluded before merge comparisons.
    if (!pbr?.baseColorTexture) continue;
    for (const ref of [pbr?.baseColorTexture, pbr?.metallicRoughnessTexture, material.normalTexture, material.occlusionTexture]) {
      if (!ref) continue;
      const source = json.textures?.[ref.index]?.source;
      if (source === undefined) return undefined;
      sources.add(source);
    }
  }
  return sources;
}

/** Preregistered before glTF parsing so DDS images never reach ImageBitmapLoader.
 * GLTFLoader still assigns texture coordinates and linear/sRGB roles normally. */
export function compressedTexturePlugin(parser: GLTFParser, pool?:SceneImagePool) {
  const json = parser.json as CompanionDocument & {images?: ImageDef[]; textures?: {source: number; sampler?: number; name?: string}[];
    samplers?: {magFilter?: number; minFilter?: number; wrapS?: number; wrapT?: number}[]};
  const sampledSources = mergeSampleImageSources(json);
  const images = new Map<number, Promise<THREE.CompressedTexture>>();
  // Match GLTFLoader's image+sampler cache. Clones with different samplers also
  // share Source identity, allowing Three's native GPU cache to reuse uploads.
  const textures = new Map<string, Promise<THREE.CompressedTexture>>();
  const loader = new DDSLoader();
  return {
    name: 'viewer_s3tc_textures',
    afterRoot() { images.clear(); textures.clear(); return null; },
    loadTexture(index: number) {
      const definition = json.textures?.[index], image = definition && json.images?.[definition.source];
      if (!definition || image?.mimeType !== 'image/vnd-ms.dds') return null;
      const key = definition.source + ':' + definition.sampler;
      const cached = textures.get(key);
      if (cached) return cached;
      let pending = images.get(definition.source);
      if (!pending) {
        pending = (parser.getDependency('bufferView', image.bufferView) as Promise<ArrayBuffer>).then(async bytes => {
          const load=()=>{
           const data=loader.parse(bytes,true);
           if((data.format!==THREE.RGB_S3TC_DXT1_Format&&data.format!==THREE.RGBA_S3TC_DXT5_Format)||!data.mipmaps.length||!data.width||!data.height)throw new Error('Invalid startup DDS image');
           return new THREE.CompressedTexture(data.mipmaps,data.width,data.height,data.format as THREE.CompressedPixelFormat);
          };
          const hash=image.extras?.viewerImageSha256;
          const identity=pool?(hash&&/^[a-f0-9]{64}$/i.test(hash)?hash.toLowerCase():await ddsDigest(bytes)):'';
          let companion:ArrayBuffer|undefined,companionIdentity='none';
          if(pool&&image.extras?.viewerCpuBufferView!==undefined){
           const originalHash=image.extras.viewerCpuImageSha256;
           companionIdentity=originalHash&&/^[a-f0-9]{64}$/i.test(originalHash)?originalHash.toLowerCase():await ddsDigest(companion=await parser.getDependency('bufferView',image.extras.viewerCpuBufferView) as ArrayBuffer);
          }
          // Equal lossy DDS bytes can originate from slightly different PNGs.
          // Include exact companion identity so tint sampling never borrows a
          // neighboring cell's different original image.
          const base=pool?await sharedDdsTexture(parser,pool,`dds:${identity}:${ddsMetadata(bytes)}:cpu:${image.extras?.viewerCpuMimeType}:${companionIdentity}`,load):load();
          if(image.extras?.viewerCpuBufferView!==undefined&&(!sampledSources||sampledSources.has(definition.source))){
           const original=companion??await parser.getDependency('bufferView',image.extras.viewerCpuBufferView) as ArrayBuffer;
           const mimeType=image.extras.viewerCpuMimeType;
           const decode=()=>createImageBitmap(new Blob([original],{type:mimeType}),{premultiplyAlpha:'none' as const,colorSpaceConversion:'none' as const});
           if(pool){
            // Check ownership again: a stream error can happen while awaiting
            // its trailing encoded companion buffer.
            registerParserImageResource(parser,base.source,{source:()=>base.source,release:()=>releaseParserCpuImage(parser,base.source)});
            addParserCpuImage(parser,base.source,decode);
           }else cpuImages.set(base.source,decode);
          }
          return base;
        });
        images.set(definition.source, pending);
      }
      const texture = pending.then(base => {
        const texture = pool?new THREE.CompressedTexture([],base.image.width,base.image.height):base.clone();
        if(pool){texture.needsUpdate=true;texture.source=base.source;texture.mipmaps=base.mipmaps;texture.format=base.format;texture.type=base.type;texture.anisotropy=pool.anisotropy;}
        const sampler = json.samplers?.[definition.sampler ?? -1] ?? {};
        texture.flipY = false;
        texture.generateMipmaps = false;
        texture.magFilter = (filters[sampler.magFilter ?? 9729] ?? THREE.LinearFilter) as THREE.MagnificationTextureFilter;
        texture.minFilter = filters[sampler.minFilter ?? 9987] ?? THREE.LinearMipmapLinearFilter;
        texture.wrapS = wraps[sampler.wrapS ?? 10497] ?? THREE.RepeatWrapping;
        texture.wrapT = wraps[sampler.wrapT ?? 10497] ?? THREE.RepeatWrapping;
        texture.name = definition.name ?? '';
        texture.userData.mimeType = image.mimeType;
        if(!pool)texture.needsUpdate = true;
        parser.associations.set(texture, {textures: index});
        return texture;
      });
      textures.set(key, texture);
      return texture;
    },
  };
}

/** Release encoded companions after merge, including maps that were never
 * sampled. Decoded bitmaps close immediately in readTextureCpuImage; the DDS
 * mip buffers on live textures remain available for WebGL context restoration. */
export function releaseTextureCpuImages(materials: Iterable<THREE.Material>, parser?:GLTFParser) {
  let released = 0;
  if(parser){const sources=parserCpuImages.get(parser);if(sources)for(const source of sources){releaseParserCpuImage(parser,source);released++;}}
  for (const material of materials) for (const value of Object.values(material)) {
    if (value instanceof THREE.Texture && cpuImages.delete(value.source)) released++;
  }
  return released;
}
