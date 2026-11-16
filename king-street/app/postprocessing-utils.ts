import type * as THREE from 'three';
import type {FullScreenQuad} from 'three/addons/postprocessing/Pass.js';

/** These shaders write every pixel and never discard. Clearing their targets
 * first only repeats the write; preserve caller state for scene/depth passes.
 */
export function renderPostQuad(renderer:THREE.WebGLRenderer,target:THREE.WebGLRenderTarget|null,quad:FullScreenQuad){
 const autoClear=renderer.autoClear;
 renderer.autoClear=false;
 try{renderer.setRenderTarget(target);quad.render(renderer);}finally{renderer.autoClear=autoClear;}
}

/** Unit normal packed into two normalized channels; folds the negative-z
 * hemisphere with a nonzero sign so all six axis directions round-trip.
 */
export const OCTAHEDRAL_NORMAL_GLSL=/* glsl */`
vec2 packOctNormal(vec3 n){
 n/=max(abs(n.x)+abs(n.y)+abs(n.z),1e-8);
 vec2 e=n.xy;
 if(n.z<0.)e=(1.-abs(e.yx))*(step(vec2(0.),e)*2.-1.);
 return e*.5+.5;
}
vec3 unpackOctNormal(vec2 e){
 vec2 f=e*2.-1.;vec3 n=vec3(f,1.-abs(f.x)-abs(f.y));
 float t=clamp(-n.z,0.,1.);
 n.xy+=vec2(n.x>=0.?-t:t,n.y>=0.?-t:t);
 return normalize(n);
}
`;
