import * as THREE from 'three';
import {ROADFIELD_GLSL} from './roadfield';

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
float gdHeight;float gdRough;
`;

const ASPHALT = /* glsl */`
float gdBand(float x,float c,float w){return 1.-smoothstep(w*.5,w,abs(x-c));}
vec3 gdAsphalt(vec3 albedo){
 vec2 p=vGdPos.xz;vec2 en=vec2(p.x,-p.y);
 float age=gdFbm(p*.06+7.3);                                   // older (greyer, lighter) or newer (darker) surfacing, ~15 m
 float blotch=gdFbm(p*.25+1.7);                                 // stains and wear, ~4 m
 vec3 c=albedo*mix(vec3(.78,.78,.8),vec3(1.42,1.38,1.3),smoothstep(.3,.7,age))*mix(.86,1.12,blotch);
 // the road field (roadfield.ts): which way the road runs, how far the kerb is, how wide the road is
 vec4 rf=rfAt(vGdPos);float on=smoothstep(.2,.6,rf.y);float d=rf.x;
 vec2 g=rfAcross(vGdPos,d);vec2 t=vec2(-g.y,g.x);if(dot(t,vec2(1.,.37))<0.)t=-t;   // one sense along the road
 float s=dot(en,t),hw=max(rf.y,.5),w=2.*hw;
 float lanes=max(1.,floor(w/3.3+.35)),lw=w/lanes,q=fract(d/lw);
 float laneLike=on*smoothstep(.5,.9,rf.z)*smoothstep(1.2,2.,hw);      // along a street, not in an intersection
 // wheel paths: two polished, rubber-darkened bands per lane, broken up along the road
 float wheel=(gdBand(q,.24,.13)+gdBand(q,.76,.13))*laneLike*mix(.45,1.,gdNoise(vec2(s*.07,d*.9)));
 c*=1.-.16*wheel;
 // oil drips down the middle of each lane: dark smudges stretched along it
 float oil=gdBand(q,.5,.11)*laneLike*smoothstep(.55,.85,gdNoise(vec2(s*.45,q*9.+3.)))*smoothstep(.35,.6,gdNoise(vec2(s*.05,7.)));
 c*=1.-.3*oil;
 // the gutter: grit, leaves and run-off darken the last half metre before the kerb
 float gutter=on*(1.-smoothstep(.15,.75,d));c*=1.-.25*gutter*mix(.6,1.,gdNoise(p*1.7));
 // utility cuts: about one 8 m cell in five carries a rectangle, mostly of dark new asphalt, now and then of paler fill
 vec2 cell=floor(p/8.);float pick=gdHash(cell+3.7);float cut=0.;
 if(pick<.2){vec2 qq=fract(p/8.)*8.;vec2 lo=vec2(.8)+gdHash2(cell)*3.,hi=lo+vec2(1.6)+gdHash2(cell+1.9)*3.4;
  cut=step(lo.x,qq.x)*step(qq.x,hi.x)*step(lo.y,qq.y)*step(qq.y,hi.y);c*=mix(1.,pick<.04?1.25:.7,cut);}
 // cracks, sealed with tar: across the road every 6-14 m in older stretches (wandering, not always kerb to kerb); plain
 // wandering cracks where the road is not lane-like
 float older=smoothstep(.42,.62,age)*(1.-cut);
 float cs=s/9.;float ci=floor(cs);float cpos=.2+.6*gdHash(vec2(ci,4.1));
 float wob=(gdNoise(vec2(d*.35,ci*7.1))-.5)*1.1+(gdNoise(vec2(d*2.3,ci*3.3))-.5)*.18;
 float xd=abs((fract(cs)-cpos)*9.-wob);
 float across=step(gdHash(vec2(ci,9.7)),.6)*smoothstep(.25,.45,gdNoise(vec2(d*.5,ci*1.9)))*laneLike;
 float crack=across*(1.-smoothstep(.006,.02,xd))*older;
 float seal=across*(1.-smoothstep(.03,.07,xd))*older;                    // the tar sealant band either side of the crack
 float wander=(1.-smoothstep(.0,.022,gdEdge(p*.45+vec2(gdNoise(p*.9)*.6))))*smoothstep(.6,.72,gdFbm(p*.035+5.1))*older*(1.-laneLike);
 c*=1.-.35*seal*gdKeep(8.);c*=1.-.55*max(crack,wander)*gdKeep(20.);
 // aggregate: stones of a few millimetres to two centimetres, the lighter ones showing where the binder has worn off
 float grain=gdNoise(p*9.)*.55+gdNoise(p*23.)*.45;float stone=gdNoise(p*61.);
 c*=mix(1.,.86+.28*grain,gdKeep(23.));c*=mix(1.,.9+.24*smoothstep(.55,.9,stone)*(1.-.6*wheel),gdKeep(61.));
 gdHeight=(grain*.6+gdNoise(p*3.1)*.4)*gdKeep(9.)+stone*.35*gdKeep(61.)-max(crack,wander)*.25-seal*.05;
 gdRough=clamp(.82-.18*wheel-.22*seal+.08*gutter,.45,.95);
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

const GARDEN = /* glsl */`
vec3 gdGarden(vec3 albedo){
 vec2 p=vGdPos.xz;
 // what the bare terrain shows is mostly back gardens: lawn, darker planting beds, rectangular patios and bare soil
 float kind=gdFbm(p*.11+3.3),wear=gdFbm(p*.6+8.1),bed=gdFbm(p*.22+6.6);
 vec3 lawn=vec3(.15,.21,.09)*mix(.7,1.12,gdFbm(p*.35+1.)),planting=vec3(.07,.11,.05)*mix(.8,1.2,wear),soil=vec3(.25,.19,.13)*mix(.8,1.15,wear);
 vec3 c=mix(lawn,planting,smoothstep(.56,.62,bed));c=mix(c,soil,smoothstep(.3,.27,kind));
 // patios and decks: about one 9 m cell in five holds a rectangle of paving 3-6 m across, squared to the street grid
 vec2 cell=floor(p/9.);float pick=gdHash(cell+11.3);float paved=smoothstep(.66,.69,kind);
 if(pick<.2){vec2 q=fract(p/9.)*9.;vec2 lo=vec2(.4)+gdHash2(cell+2.)*2.,hi=lo+vec2(3.)+gdHash2(cell+5.)*3.;
  paved=max(paved,step(lo.x,q.x)*step(q.x,hi.x)*step(lo.y,q.y)*step(q.y,hi.y));}
 c=mix(c,albedo*mix(.82,1.04,wear),paved);
 c*=mix(1.,.86+.28*gdNoise(p*12.),gdKeep(12.));
 gdHeight=(1.-paved)*gdNoise(p*6.)*gdKeep(6.);
 return c;
}
`;

const PAVEMENT = /* glsl */`
vec3 gdPavement(vec3 albedo){
 vec2 p=vGdPos.xz;
 // a sidewalk is laid in pours of different ages and wears unevenly: tone drift over metres, weathered blotches, and
 // the dark spots of old chewing gum, densest nowhere in particular
 float tone=gdFbm(p*.3+4.2),blot=gdFbm(p*1.1+9.9);
 vec3 c=albedo*mix(.88,1.07,tone)*mix(1.,.9,smoothstep(.62,.8,blot));
 vec2 g=floor(p*2.);float spot=step(.985,gdHash(g+1.3));vec2 o=fract(p*2.)-.5+(gdHash2(g)-.5)*.5;
 c*=1.-.45*spot*(1.-smoothstep(.04,.07,length(o)))*gdKeep(4.);
 gdHeight=gdNoise(p*7.)*gdKeep(7.)*.5;
 return c;
}
`;

function patch(materials: Iterable<THREE.MeshStandardMaterial>, tag: string, body: string, call: string, bump: number, roughness: string, uniforms: Record<string, THREE.IUniform> = {}, extra = '') {
  for (const m of materials) {
    if (m.userData.groundDetail) continue; m.userData.groundDetail = tag;
    const previous = m.onBeforeCompile.bind(m);
    m.onBeforeCompile = (shader, renderer) => {
      previous(shader, renderer); Object.assign(shader.uniforms, uniforms);
      shader.vertexShader = shader.vertexShader.replace('#include <common>', '#include <common>\nvarying vec3 vGdPos;')
        .replace('#include <project_vertex>', '#include <project_vertex>\nvGdPos=(modelMatrix*vec4(transformed,1.)).xyz;');
      shader.fragmentShader = shader.fragmentShader.replace('#include <common>', '#include <common>\n' + COMMON + extra + body)
        .replace('#include <map_fragment>', `#include <map_fragment>\ndiffuseColor.rgb=${call}(diffuseColor.rgb);`)
        .replace('#include <roughnessmap_fragment>', `#include <roughnessmap_fragment>\nroughnessFactor=${roughness};`)
        .replace('#include <normal_fragment_maps>', `#include <normal_fragment_maps>\nnormal=gdBump(normal,gdHeight,${bump.toFixed(4)});`);
    };
    const key = m.customProgramCacheKey.bind(m);
    m.customProgramCacheKey = () => key() + '|ground-' + tag;
    m.needsUpdate = true;
  }
}

/** Wear, lane wheel paths, oil, gutters, patches, sealed cracks and aggregate on road asphalt. `field` holds the road
 *  field's uniforms (roadfield.ts); its texture may be set after this call, before the first frame. */
export function detailAsphalt(materials: Iterable<THREE.MeshStandardMaterial>, field: Record<string, THREE.IUniform>) {
  patch(materials, 'asphalt2', ASPHALT, 'gdAsphalt', .006, 'gdRough', field, ROADFIELD_GLSL);
}

/** Lush and dry patches, mowing stripes, blade grain and relief on lawns. */
export function detailGrass(materials: Iterable<THREE.MeshStandardMaterial>) {
  patch(materials, 'grass', GRASS, 'gdGrass', .012, 'clamp(max(roughnessFactor,.9),.0,1.)');
}

/** Back gardens, patios and bare soil where the terrain itself shows (its material is one paving texture). */
export function detailGarden(materials: Iterable<THREE.MeshStandardMaterial>) {
  patch(materials, 'garden', GARDEN, 'gdGarden', .01, 'clamp(max(roughnessFactor,.82),.0,1.)');
}

/** Tone drift, weathering and gum spots on sidewalk concrete (the slab joints are their own geometry). */
export function detailPavement(materials: Iterable<THREE.MeshStandardMaterial>) {
  patch(materials, 'pavement', PAVEMENT, 'gdPavement', .004, 'clamp(roughnessFactor*mix(1.05,.92,gdHeight),.4,1.)');
}

/** Ground classes this module details, by a mesh's material class. */
export const ASPHALT_CLASS = /^asphalt/, GRASS_CLASS = /^(grass|turf|lawn)/;
