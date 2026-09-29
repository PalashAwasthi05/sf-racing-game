import * as THREE from 'three';

// Windows: glazing in the generated world is an opaque dark surface, so a facade seen face-on was a grid of black
// holes. Real windows in daylight show a dim room behind a reflection that strengthens toward grazing angles. The
// glass here shows a room behind it: the view ray is followed into a box of room size (4 m bays, 3.6 m storeys, 6 m deep)
// laid out along the facade in world space, and the wall, floor or ceiling it meets is drawn at interior brightness
// (a tenth of daylight; a few rooms lit), weighted by the glass's own Fresnel so the reflection still takes over at
// grazing angles. The room grid is generic, not the building's measured storeys. Skylights and the insides of
// buildings are left as they were.

const HEADER = /* glsl */`
varying vec3 vGlPos;varying vec3 vGlNrm;
vec3 glInterior(){
 vec3 N=normalize(vGlNrm);if(abs(N.y)>.5)return vec3(0.);
 vec3 V=normalize(vGlPos-cameraPosition);float facing=-dot(V,N);if(facing<=.02)return vec3(0.);
 vec3 T=normalize(vec3(-N.z,0.,N.x));
 vec3 o=vec3(dot(vGlPos.xz,T.xz),vGlPos.y,0.);vec3 d=vec3(dot(V.xz,T.xz),V.y,facing);
 vec3 cell=vec3(4.,3.6,6.);vec2 id=floor(o.xy/cell.xy);
 vec3 lo=vec3(id*cell.xy,0.),hi=vec3((id+1.)*cell.xy,cell.z);
 vec3 dd=sign(d)*max(abs(d),vec3(1e-4));vec3 tA=(lo-o)/dd,tB=(hi-o)/dd;vec3 tx=max(tA,tB);
 float t=min(min(tx.x,tx.y),tx.z);vec3 h=o+dd*t;
 float r=fract(sin(dot(id,vec2(12.9898,78.233)))*43758.5453);float r2=fract(r*17.37);
 vec3 wallC=mix(vec3(.62,.58,.52),vec3(.5,.53,.56),r2);
 vec3 c;float lit=step(.7,r);
 if(t==tx.z)c=wallC*.85;
 else if(t==tx.y){if(dd.y>0.){c=vec3(.82,.8,.76);c*=1.+lit*3.*step(.72,fract(h.x*.5))*step(.5,fract(h.z*.33));}else c=vec3(.3,.26,.22)*mix(.8,1.2,r2);}
 else c=wallC*.7;
 float fade=exp(-t*.14);
 vec3 L=c*mix(.35,1.,fade)*(.022+.075*lit);
 float F=.04+.96*pow(1.-clamp(facing,0.,1.),5.);
 return L*(1.-F);
}
`;

const GLASS = /^(glass|glass_dark|glass_light|glass_blue|glass_clear)$/;
export function isGlazing(cls: string) { return GLASS.test(cls); }

/** Draws a room behind each given glazing material (chained after any existing onBeforeCompile). */
export function glazeMaterials(materials: Iterable<THREE.MeshStandardMaterial>) {
  for (const m of materials) {
    if (m.userData.glazed) continue; m.userData.glazed = true;
    // the glass has no diffuse of its own: sunlight on it is the reflection, not a lit coloured surface
    m.color.multiplyScalar(.25);
    const previous = m.onBeforeCompile.bind(m);
    m.onBeforeCompile = (shader, renderer) => {
      previous(shader, renderer);
      shader.vertexShader = shader.vertexShader.replace('#include <common>', '#include <common>\nvarying vec3 vGlPos;varying vec3 vGlNrm;')
        .replace('#include <project_vertex>', '#include <project_vertex>\nvGlPos=(modelMatrix*vec4(transformed,1.)).xyz;vGlNrm=normalize(mat3(modelMatrix)*objectNormal);');
      shader.fragmentShader = shader.fragmentShader.replace('#include <common>', '#include <common>\n' + HEADER)
        .replace('#include <emissivemap_fragment>', '#include <emissivemap_fragment>\ntotalEmissiveRadiance+=glInterior();');
    };
    const key = m.customProgramCacheKey.bind(m);
    m.customProgramCacheKey = () => key() + '|glazed';
    m.needsUpdate = true;
  }
}
