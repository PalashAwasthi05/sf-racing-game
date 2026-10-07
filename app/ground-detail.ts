import * as THREE from 'three';

// Ground detail: what makes a road and a lawn read as worn and grown rather than painted. The generated world gives the
// road one asphalt texture repeating every few metres and the lawns one flat green. Here, in world space:
//  asphalt  tone drifting between older, greyer surfacing and newer, darker patches over tens of metres; rectangular
//           utility-cut patches; sealed cracks in some stretches; a fine aggregate speckle; a little relief.
//  grass    dry and lush patches, mowing stripes, blade-scale grain and relief.
// Every fine pattern fades with its size on screen, so a lawn seen at a grazing angle does not shimmer. Nothing here is
// location specific: a material is matched by its class name alone.

const COMMON = /* glsl */`
varying vec3 vGdPos;
float gdHash(vec2 p){return fract(sin(dot(p,vec2(127.1,311.7)))*43758.5453);}
vec2 gdHash2(vec2 p){return fract(sin(vec2(dot(p,vec2(127.1,311.7)),dot(p,vec2(269.5,183.3))))*43758.5453);}
float gdNoise(vec2 p){vec2 i=floor(p),f=fract(p);f=f*f*(3.-2.*f);return mix(mix(gdHash(i),gdHash(i+vec2(1.,0.)),f.x),mix(gdHash(i+vec2(0.,1.)),gdHash(i+vec2(1.,1.)),f.x),f.y);}
float gdFbm(vec2 p){float v=0.,a=.5;for(int i=0;i<4;i++){v+=a*gdNoise(p);p=p*2.03+vec2(17.,9.);a*=.5;}return v;}
// the share of a pattern of frequency f (cycles per metre) that survives at this pixel's footprint
float gdKeep(float f){vec2 w=fwidth(vGdPos.xz);return 1.-smoothstep(.25,.5,max(w.x,w.y)*f);}
// distance to the nearest edge of a jittered cell pattern (cracks)
float gdEdge(vec2 p){vec2 i=floor(p),f=fract(p);float d1=8.,d2=8.;
 for(int y=-1;y<=1;y++)for(int x=-1;x<=1;x++){vec2 o=vec2(float(x),float(y));vec2 r=o+gdHash2(i+o)-f;float d=dot(r,r);if(d<d1){d2=d1;d1=d;}else if(d<d2)d2=d;}
 return sqrt(d2)-sqrt(d1);}
// a bumped normal from a height field h (derivative bump mapping)
vec3 gdBump(vec3 n,float h,float k){vec3 dpx=dFdx(-vViewPosition),dpy=dFdy(-vViewPosition);float hx=dFdx(h),hy=dFdy(h);
 vec3 r1=cross(dpy,n),r2=cross(n,dpx);float det=dot(dpx,r1);vec3 g=sign(det)*(hx*r1+hy*r2);return normalize(abs(det)*n-k*g);}
float gdHeight;
`;

const ASPHALT = /* glsl */`
vec3 gdAsphalt(vec3 albedo){
 vec2 p=vGdPos.xz;
 float age=gdFbm(p*.06+7.3);                                   // older (greyer, lighter) or newer (darker) surfacing, ~15 m
 float blotch=gdFbm(p*.25+1.7);                                 // stains and wear, ~4 m
 vec3 c=albedo*mix(vec3(.72,.72,.75),vec3(1.5,1.44,1.34),smoothstep(.3,.7,age))*mix(.84,1.14,blotch);
 // utility cuts: about one 8 m cell in four carries a rectangle, of dark new asphalt or of pale fill
 vec2 cell=floor(p/8.);float pick=gdHash(cell+3.7);
 if(pick<.26){vec2 q=fract(p/8.)*8.;vec2 lo=vec2(.8)+gdHash2(cell)*3.,hi=lo+vec2(1.6)+gdHash2(cell+1.9)*3.4;
  float inside=step(lo.x,q.x)*step(q.x,hi.x)*step(lo.y,q.y)*step(q.y,hi.y);c*=mix(1.,pick<.09?1.75:.62,inside);}
 // sealed cracks in the older stretches
 float crack=1.-smoothstep(.0,.035,gdEdge(p*.38));c*=1.-.45*crack*smoothstep(.45,.7,age)*gdKeep(10.);
 // aggregate speckle and a fine relief
 float grain=gdNoise(p*9.)*.6+gdNoise(p*23.)*.4;c*=mix(1.,.9+.2*grain,gdKeep(23.));
 gdHeight=(grain*.6+gdNoise(p*3.1)*.4)*gdKeep(9.)-crack*.5;
 return c;
}
`;

const GRASS = /* glsl */`
vec3 gdGrass(vec3 albedo){
 vec2 p=vGdPos.xz;
 float lush=gdFbm(p*.06+2.1),dry=smoothstep(.55,.85,gdFbm(p*.11+9.4));
 vec3 c=albedo*mix(vec3(.78,.86,.7),vec3(1.12,1.18,1.02),lush);
 c=mix(c,c*vec3(1.35,1.18,.78),.55*dry);                          // dry, straw-coloured patches
 c*=1.+.06*sign(sin(dot(p,vec2(.6,.8))*2.62))*gdKeep(.42);          // mowing stripes, 1.2 m wide
 float blade=gdNoise(p*31.)*.55+gdNoise(p*77.)*.45;
 c*=mix(1.,.82+.36*blade,gdKeep(31.));
 c*=mix(1.,.9+.2*gdNoise(p*1.3),1.);                               // clumps
 gdHeight=blade*gdKeep(31.)+gdNoise(p*4.)*.5;
 return c;
}
`;

function patch(materials: Iterable<THREE.MeshStandardMaterial>, tag: string, body: string, call: string, bump: number, roughness: string) {
  for (const m of materials) {
    if (m.userData.groundDetail) continue; m.userData.groundDetail = tag;
    const previous = m.onBeforeCompile.bind(m);
    m.onBeforeCompile = (shader, renderer) => {
      previous(shader, renderer);
      shader.vertexShader = shader.vertexShader.replace('#include <common>', '#include <common>\nvarying vec3 vGdPos;')
        .replace('#include <project_vertex>', '#include <project_vertex>\nvGdPos=(modelMatrix*vec4(transformed,1.)).xyz;');
      shader.fragmentShader = shader.fragmentShader.replace('#include <common>', '#include <common>\n' + COMMON + body)
        .replace('#include <map_fragment>', `#include <map_fragment>\ndiffuseColor.rgb=${call}(diffuseColor.rgb);`)
        .replace('#include <roughnessmap_fragment>', `#include <roughnessmap_fragment>\nroughnessFactor=${roughness};`)
        .replace('#include <normal_fragment_maps>', `#include <normal_fragment_maps>\nnormal=gdBump(normal,gdHeight,${bump.toFixed(4)});`);
    };
    const key = m.customProgramCacheKey.bind(m);
    m.customProgramCacheKey = () => key() + '|ground-' + tag;
    m.needsUpdate = true;
  }
}

/** Wear, patches, cracks and relief on road asphalt. */
export function detailAsphalt(materials: Iterable<THREE.MeshStandardMaterial>) {
  patch(materials, 'asphalt', ASPHALT, 'gdAsphalt', .006, 'clamp(roughnessFactor*mix(1.04,.9,step(.5,gdHeight)),.3,1.)');
}

/** Lush and dry patches, mowing stripes, blade grain and relief on lawns. */
export function detailGrass(materials: Iterable<THREE.MeshStandardMaterial>) {
  patch(materials, 'grass', GRASS, 'gdGrass', .012, 'clamp(max(roughnessFactor,.9),.0,1.)');
}

/** Ground classes this module details, by a mesh's material class. */
export const ASPHALT_CLASS = /^asphalt/, GRASS_CLASS = /^(grass|turf|lawn)/;
