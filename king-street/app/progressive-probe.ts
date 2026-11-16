import type * as THREE from 'three';

type Blur = (target: THREE.WebGLRenderTarget, from: number, to: number, sigma: number, axis: THREE.Vector3) => void;

/** Three r180 schedules all roughness levels synchronously. Keep its actual
 * kernels, axes and packing, but spread the private working atlas across frames.
 * Only a completed atlas can be returned to the street materials. */
export class ProgressiveProbe {
  private readonly filter: THREE.PMREMGenerator & { _blur?: Blur };
  private levels: Parameters<Blur>[] = [];
  private output: THREE.WebGLRenderTarget | null = null;
  private cursor = 0;
  constructor(filter: THREE.PMREMGenerator) { this.filter = filter; }
  get pending() { return this.output !== null; }

  start(cube: THREE.CubeTexture, spare: THREE.WebGLRenderTarget) {
    this.cancel();
    const blur = this.filter._blur;
    if (typeof blur !== 'function') {
      this.output = this.filter.fromCubemap(cube, spare);
      return;
    }
    this.filter._blur = (...args) => { this.levels.push(args); };
    try { this.output = this.filter.fromCubemap(cube, spare); }
    finally { this.filter._blur = blur; }
  }

  advance(renderer: THREE.WebGLRenderer): THREE.WebGLRenderTarget | null {
    const target = this.output;
    if (!target) return null;
    const previous = renderer.getRenderTarget(), face = renderer.getActiveCubeFace(), mip = renderer.getActiveMipmapLevel();
    const clear = renderer.autoClear, xr = renderer.xr.enabled;
    try {
      renderer.autoClear = false;
      renderer.xr.enabled = false;
      target.scissorTest = true;
      // The first two reductions process most of the texels. Later levels cover
      // at most 32 pixels per face and can share one frame.
      const end = this.cursor < 2 ? this.cursor + 1 : this.levels.length;
      for (; this.cursor < Math.min(end, this.levels.length); this.cursor++) this.filter._blur!(...this.levels[this.cursor]);
    } finally {
      renderer.autoClear = clear;
      renderer.xr.enabled = xr;
      target.scissorTest = false;
      target.viewport.set(0, 0, target.width, target.height);
      target.scissor.copy(target.viewport);
      renderer.setRenderTarget(previous, face, mip);
    }
    if (this.cursor < this.levels.length) return null;
    this.cancel();
    return target;
  }

  cancel() { this.levels.length = 0; this.cursor = 0; this.output = null; }
}
