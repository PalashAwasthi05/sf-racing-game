import * as THREE from 'three';
import {isPooledImageSource} from './scene-image-pool.ts';

function textures(material: THREE.Material) {
  return Object.values(material).filter((value): value is THREE.Texture => value instanceof THREE.Texture);
}

/** Texture.dispose only frees the GPU object. glTF ImageBitmaps and parser
 * associations otherwise keep decoded retired variants alive indefinitely.
 * Shared Sources and shared image objects used by any active material survive. */
export function retireMaterialResources(retired: THREE.Material[], active: Iterable<THREE.Material>, associations?: Map<unknown, unknown>) {
  const activeMaterials = new Set(active);
  const kept = new Set<THREE.Texture>(), sources = new Set<THREE.Source>(), images = new Set<unknown>();
  const imageObjects = (image: unknown): unknown[] => Array.isArray(image) ? image.flatMap(imageObjects) : [image];
  for (const material of activeMaterials) for (const texture of textures(material)) {
    kept.add(texture); sources.add(texture.source);
    for (const image of imageObjects(texture.image)) images.add(image);
  }
  const unused = new Set<THREE.Texture>(), closed = new Set<unknown>();
  for (const material of new Set(retired)) {
    if (activeMaterials.has(material)) continue;
    for (const texture of textures(material)) if (!kept.has(texture)) unused.add(texture);
    associations?.delete(material);
    material.dispose();
  }
  for (const texture of unused) {
    texture.dispose();
    associations?.delete(texture);
    // CompressedTexture stores DDS bytes on mipmaps rather than Source.data.
    // Assign a new array: active sampler clones may share the old mip chain.
    if ((texture as THREE.CompressedTexture).isCompressedTexture) texture.mipmaps = [];
    if (sources.has(texture.source) || isPooledImageSource(texture.source)) continue;
    for (const image of imageObjects(texture.image)) {
      if (!image || images.has(image) || closed.has(image)) continue;
      const bitmap = image as {close?: () => void};
      if (typeof bitmap.close === 'function') bitmap.close();
      closed.add(image);
    }
    // Also drops HTML image/canvas references and data held by parser source caches.
    texture.source.data = null;
  }
  return {textures: unused.size, images: closed.size};
}

/** Final cleanup for textures created by one viewer's loaders. ImageLoader may
 * share cached images between independent Texture/Source objects: never close
 * those backing images, or a concurrent viewer can lose its upload source. */
export function disposeOwnedTextures(textures: Iterable<THREE.Texture>) {
  const cached = new Set<unknown>(Object.values(THREE.Cache.files));
  const sources = new Set<THREE.Source>(), closed = new Set<unknown>();
  for (const texture of new Set(textures)) {
    if (cached.has(texture) || cached.has(texture.source)) continue;
    texture.dispose();
    if ((texture as THREE.CompressedTexture).isCompressedTexture) texture.mipmaps = [];
    if (sources.has(texture.source) || isPooledImageSource(texture.source)) continue;
    sources.add(texture.source);
    const images = Array.isArray(texture.image) ? texture.image.flat(Infinity) : [texture.image];
    for (const image of images) {
      if (!image || cached.has(image) || closed.has(image)) continue;
      if (typeof image.close === 'function') { image.close(); closed.add(image); }
    }
    texture.source.data = null;
  }
}
