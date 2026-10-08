import * as THREE from 'three';
import {ROADFIELD_GLSL} from './roadfield';
import {PAVING_GLSL} from './paving';

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
 // where the terrain shows in the narrow gaps between a road and its kerb, or along a building's foot, it is the
 // gutter's or the wall foot's grime, not lawn (roadfield.ts: distance from the road's edge and from the nearest wall)
 vec4 rf=rfAt(vGdPos);float gap=max(1.-smoothstep(.45,.9,abs(rf.x)),1.-smoothstep(.5,1.,rf.w));
 c=mix(c,albedo*vec3(.42,.41,.39)*mix(.8,1.1,wear),gap);paved=max(paved,gap);
 c*=mix(1.,.86+.28*gdNoise(p*12.),gdKeep(12.));
 gdHeight=(1.-paved)*gdNoise(p*6.)*gdKeep(6.);
 return c;
}
`;

const PAVEMENT = /* glsl */`
// the share of a pixel (px metres wide) covered by a line of half width hw at signed distance x: a box filter, so a thin
// line far away greys the pixel by its true share instead of shimmering
float gdLine(float x,float hw,float px){return max(0.,min(x+.5*px,hw)-max(x-.5*px,-hw))/px;}
vec3 gdPavement(vec3 albedo){
 vec2 p=vGdPos.xz;
 float level=smoothstep(.6,.8,abs(normalize(cross(dFdx(vGdPos),dFdy(vGdPos))).y));   // the walk, not the kerb's face
 // where this point stands: metres from the kerb's edge, and from the nearest building wall (roadfield.ts)
 vec4 rf=rfAt(vGdPos);float dk=max(-rf.x,0.),db=rf.w;
 // the slabs: the generator's joint runs where they reach (paving.ts), else a layout along the kerb of its own
 vec4 pv=pfAt(vGdPos);bool run=pv.w>.1;vec2 a;float ph,rid,ks,span;
 if(run){
  a=vec2(cos(pv.x),sin(pv.x));ph=pv.y;rid=fract(sin(pv.z*1.913+pv.x*7.31)*43758.5);
  // across the run in its own straight frame: the strips reach from the kerb to their far end, which is the end the
  // kerb distance grows toward
  vec2 n=vec2(-a.y,a.x);float v=dot(p,n)-pv.z,hl=pv.w;vec3 n3=vec3(n.x,0.,n.y);
  float s=-rfAt(vGdPos+n3*(hl-v)).x>-rfAt(vGdPos-n3*(hl+v)).x?1.:-1.;
  ks=max(hl+s*v,0.);span=2.*hl;
 }else{
  // no run: square slabs along the kerb. The direction is snapped to 5 degree steps, so it is one constant along a
  // straight kerb: a direction that turned a little from pixel to pixel would swing dot(p, along) by metres this far
  // from the origin and bend every joint into loops
  vec2 g=rfAcross(vGdPos,rf.x);float ang=atan(g.x,g.y);ang=floor(ang/.0872665+.5)*.0872665;
  a=vec2(cos(ang),sin(ang));ph=0.;rid=.37+ang;ks=dot(p,vec2(-a.y,a.x));span=0.;
 }
 float t=dot(p,a)-ph,k=floor(t/1.8),ft=t-k*1.8;
 // across the walk: the generator's slabs span its strips' length from the kerb; a wider walk is scored in 1.6 m bands
 float band=ks<span?0.:1.+floor((ks-span)/1.6),fb=ks<span?ks/span:fract((ks-span)/1.6);
 if(!run){float q=dot(p,vec2(-a.y,a.x))/1.6;band=floor(q);fb=fract(q);}
 vec2 sid=vec2(k+floor(rid*997.)*61.,band);
 float r1=gdHash(sid+.17),r2=gdHash(sid*1.31+4.7),r3=gdHash(sid*.71+9.3),r4=gdHash(sid*2.03+1.9);
 // pours of different ages: most slabs within a few percent of each other, one in fourteen replaced (paler, cleaner,
 // cooler) and one in sixteen stained darker; over tens of metres the whole walk drifts in tone
 bool fresh=r2<.07,stained=r2>.94;
 vec3 c=albedo*vec3(1.03,1.,.95)*(.86+.24*r1)*mix(.93,1.05,gdFbm(p*.3+4.2));
 if(fresh)c=albedo*vec3(1.13,1.15,1.19);
 if(stained)c*=.74+.1*gdNoise(p*2.+r3*9.);
 // grime by the kerb and dirt banked against the frontage; the walking line between is kept cleaner by feet
 float kerbZone=1.-smoothstep(0.,.9,dk),wallZone=1.-smoothstep(0.,.7,db);
 float walk=smoothstep(1.,1.8,dk)*smoothstep(.5,1.4,db);
 c*=1.-.24*(1.-smoothstep(.05,.45,dk))*mix(.7,1.,gdNoise(p*2.1));          // the kerb: tyre scuffs, gutter splash
 c*=1.-.14*kerbZone*mix(.5,1.,gdNoise(p*1.3));
 c*=1.-.27*wallZone*mix(.5,1.,gdNoise(p*.9+3.));
 c*=1.+.05*walk;
 // ground-in grime: specks and smears of a few centimetres to a few decimetres, thicker away from the walking line
 float speck=smoothstep(.5,.75,gdNoise(p*9.+r1*5.)*.6+gdNoise(p*23.)*.4)*gdKeep(9.);
 float smear=smoothstep(.45,.75,gdFbm(p*3.3+1.1));
 c*=1.-(.1*speck+.08*smear)*(1.-.5*walk);
 // spills and drips: irregular dark patches, more of them toward the kerb and the frontage
 float spill=smoothstep(.66,.8,gdFbm(p*1.7+vec2(r1*3.,0.))+.12*(kerbZone+wallZone));
 c*=1.-.26*spill*(fresh?.4:1.);
 // chewing gum: dark spots of a centimetre or two, most of them in the walking line
 vec2 gq=p*4.,gi=floor(gq);float gum=0.;
 if(gdHash(gi+.3)<.03+.07*walk){vec2 o=fract(gq)-.5+(gdHash2(gi)-.5)*.6;gum=1.-smoothstep(.035,.06,length(o));}
 c*=1.-.5*gum*gdKeep(8.)*level;
 // a crack across one slab in eleven, wandering, a hairline of dirt
 float pxT=max(fwidth(t),1e-4),pxK=max(fwidth(ks),1e-4),crack=0.;
 if(r3<.09&&!fresh){float x0=.3+1.2*r4,x1=.3+1.2*fract(r4*13.7);
  float xc=mix(x0,x1,fb)+(gdNoise(vec2(fb*9.,r3*40.))-.5)*.22+(gdNoise(vec2(fb*31.,r4*20.))-.5)*.05;
  crack=gdLine(ft-xc,.0025,pxT);}
 // the joints: a tooled groove a centimetre wide with dirt worked into its edges, drawn here too so the grid holds
 // where the generator drew none (past 3.2 m from the kerb, and where no run reaches)
 float jx=min(ft,1.8-ft),jy=run&&ks<span?span-ks:min(fb,1.-fb)*1.6;
 float joint=max(gdLine(jx,.006,pxT),gdLine(jy,.006,pxK)),halo=max(gdLine(jx,.035,pxT),gdLine(jy,.035,pxK));
 // a broom finish: fine ridges along the joints, seen only close to
 float broom=gdNoise(vec2(t*140.,fb*3.+r1*9.));
 c*=mix(1.,.97+.06*broom,gdKeep(140.)*level);
 c*=1.-level*(.16*halo+.5*joint+.55*crack);
 gdHeight=gdNoise(p*7.)*gdKeep(7.)*.5+broom*.25*gdKeep(140.)-.8*joint-.5*crack;
 gdRough=1.-.1*walk+.05*spill+(fresh?.04:0.);   // a factor on the scan's own roughness
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

/** Back gardens, patios and bare soil where the terrain itself shows (its material is one paving texture); grime in the
 *  gaps it shows through along kerbs and building walls. `field` holds the road field's uniforms (roadfield.ts). */
export function detailGarden(materials: Iterable<THREE.MeshStandardMaterial>, field: Record<string, THREE.IUniform>) {
  patch(materials, 'garden2', GARDEN, 'gdGarden', .01, 'clamp(max(roughnessFactor,.82),.0,1.)', field, ROADFIELD_GLSL);
}

/** Slab by slab tone, joints, cracks, gum, spills, kerb and frontage dirt on sidewalk concrete. `fields` holds the road
 *  field's and the paving field's uniforms (roadfield.ts, paving.ts); their textures may be set after this call, before
 *  the first frame. */
export function detailPavement(materials: Iterable<THREE.MeshStandardMaterial>, fields: Record<string, THREE.IUniform>) {
  patch(materials, 'pavement2', PAVEMENT, 'gdPavement', .004, 'clamp(roughnessFactor*gdRough,.45,1.)', fields, ROADFIELD_GLSL + PAVING_GLSL);
}

/** Ground classes this module details, by a mesh's material class. */
export const ASPHALT_CLASS = /^asphalt/, GRASS_CLASS = /^(grass|turf|lawn)/;
