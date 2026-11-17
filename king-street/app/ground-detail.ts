import * as THREE from 'three';
import {ROADFIELD_GLSL} from './roadfield.ts';
import {PAVING_GLSL} from './paving.ts';

// Ground detail: what makes a road and a lawn read as worn and grown rather than painted. The generated world gives the
// road one asphalt texture repeating every few metres and the lawns one flat green. Here, in world space:
//  asphalt  tone drifting between older, greyer surfacing and newer, darker patches over tens of metres;
//           sealed cracks in some stretches; a fine aggregate speckle; a little relief.
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
 vec3 r1=cross(dpy,n),r2=cross(n,dpx);float det=dot(dpx,r1);if(abs(det)<1e-12)return n;vec3 g=sign(det)*(hx*r1+hy*r2);return normalize(abs(det)*n-k*g);}
float gdHeight;float gdRough;vec2 gdSlope;
`;

// Close-range surface layers (work/material-assets/generate_ground.py): six 1 m tiles in one array texture, one sampler.
// 0 dense asphalt, 1 worn asphalt, 2 cracked asphalt, 3 broomed concrete, 4 weathered concrete, 5 trowelled concrete;
// 6 and 7 are 8 m tiles of old asphalt and old concrete (cracks, sealant, stains) for what reads from a moving car.
// Each texel: the surface's slope along the tile's x and y (as a normal's xy) and its tone about 0.5.
const LAYER_COUNT = 8;
/** Keep the height beside the slope/tone channels sampled at identical UVs. */
export function packGroundLayers(surface:Uint8ClampedArray<ArrayBuffer>,height:Uint8ClampedArray<ArrayBuffer>|null){
  const data=new Uint8Array(surface.buffer,surface.byteOffset,surface.byteLength);
  const matches=height?.length===surface.length;
  for(let i=3;i<data.length;i+=4)data[i]=matches?height![i-3]:255;
  return data;
}
function flatLayers() {
  const data = new Uint8Array(4 * LAYER_COUNT); for (let i = 0; i < LAYER_COUNT; i++) data.set([128,128,128,255],4*i);
  const t = new THREE.DataArrayTexture(data, 1, 1, LAYER_COUNT); t.needsUpdate = true; return t;
}
const layers: {value: THREE.DataArrayTexture} = {value: flatLayers()};
/** The sun's direction (world), for the shadows the ground's own relief casts; the viewer sets it with the lighting. */
export const groundSun = {value: new THREE.Vector3(0, 1, 0)};
let layersRequested = false;
function loadLayers() {
  if (layersRequested || typeof Image === 'undefined' || typeof document === 'undefined' || typeof (document as Document).createElementNS !== 'function') return; layersRequested = true;
  // Eight RGBA layers contain slope/tone RGB and the matching height in alpha.
  const load = (url: string) => new Promise<Uint8ClampedArray<ArrayBuffer> | null>(resolve => {
    const image = new Image();
    image.onerror = () => resolve(null);
    image.onload = () => {
      const size = image.width, canvas = document.createElement('canvas'); canvas.width = size; canvas.height = size * LAYER_COUNT;
      const context = canvas.getContext('2d', {willReadFrequently: true}); if (!context || image.height !== size * LAYER_COUNT) { resolve(null); return; }
      context.drawImage(image, 0, 0); const pixels=context.getImageData(0,0,size,size*LAYER_COUNT).data;
      canvas.width=canvas.height=1;image.src='';resolve(pixels);
    };
    image.src = url;
  });
  void Promise.all([load('/materials/ground-layers.png'), load('/materials/ground-layers-height.png')]).then(([surface, height]) => {
    if (!surface) return;
    const size = Math.round(Math.sqrt(surface.length / 4 / LAYER_COUNT)), data = packGroundLayers(surface,height);
    const t = new THREE.DataArrayTexture(data, size, size, LAYER_COUNT);
    t.format = THREE.RGBAFormat; t.type = THREE.UnsignedByteType; t.colorSpace = THREE.NoColorSpace; t.wrapS = t.wrapT = THREE.RepeatWrapping;
    t.minFilter = THREE.LinearMipmapLinearFilter; t.magFilter = THREE.LinearFilter; t.generateMipmaps = true; t.anisotropy = 8; t.needsUpdate = true;
    layers.value.dispose(); layers.value = t;
  });
}
const LAYERS = /* glsl */`
uniform highp sampler2DArray uGdLayers;
// xy: slope (-1..1), z: tone. The derivatives are passed in, so a lookup whose coordinates jump (from slab to slab) or
// that sits in a branch still picks the right mip.
vec3 gdLayerAt(vec2 uv,float layer,vec2 dx,vec2 dy){vec3 t=textureGrad(uGdLayers,vec3(uv,layer),dx,dy).rgb;return vec3(t.xy*2.-1.,t.z);}
// two copies of a layer, one turned and offset, traded over a few metres: the 1 m tile shows no repeat
// shift moves the first copy's lookup (the relief's parallax)
vec3 gdLayer(vec2 p,float layer,vec2 shift){
 vec2 dx=dFdx(p),dy=dFdy(p);const mat2 R=mat2(.7986,-.6018,.6018,.7986);
 vec3 a=gdLayerAt(p+shift,layer,dx,dy),b=gdLayerAt(R*p*1.13+vec2(.37,.71),layer,R*dx*1.13,R*dy*1.13);b.xy=transpose(R)*b.xy;
 return mix(a,b,smoothstep(.35,.65,gdNoise(p*.37+layer*3.1)));
}
vec3 gdLayer(vec2 p,float layer){return gdLayer(p,layer,vec2(0.));}
// Relief: the layer's height field seen at an angle. The view ray is walked down through the heights (six steps and a
// straight-line refinement) to where it meets the surface, so a crack's far wall hides its floor and the surface slides
// against itself as the eye moves; a second, shorter walk toward the sun finds what the relief shadows.
//  uv, dx, dy  the layer coordinates and their screen derivatives      toEye, toSun  unit vectors in the layer's frame
//  depth       the relief's full depth in layer units (y up)           amount        0 turns it off (far away)
// Returns the shift to add to uv; shade is 1 in the relief's own shadow.
uniform vec3 uGdSun;
float gdHeightAt(vec2 uv,float layer,vec2 dx,vec2 dy){return textureGrad(uGdLayers,vec3(uv,layer),dx,dy).a;}
vec2 gdRelief(vec2 uv,vec2 dx,vec2 dy,float layer,vec3 toEye,vec3 toSun,float depth,float amount,out float shade){
 shade=0.;if(amount<.01)return vec2(0.);
 vec2 run=-toEye.xz/max(toEye.y,.14)*depth*amount;                 // the lookup's travel from the top of the relief to its floor
 float prevRay=1.,prevH=gdHeightAt(uv,layer,dx,dy),t=0.;
 for(int i=1;i<=6;i++){
  float ray=1.-float(i)/6.,h=gdHeightAt(uv+run*(1.-ray),layer,dx,dy);
  if(h>=ray){float a=prevRay-prevH,b=h-ray;t=1.-mix(ray,prevRay,b/max(a+b,1e-4));break;}
  prevRay=ray;prevH=h;t=1.;
 }
 vec2 shift=run*t;float h0=1.-t;
 if(toSun.y>.03){
  vec2 climb=toSun.xz/toSun.y*depth*amount;                        // travel toward the sun per unit of height
  for(int i=1;i<=3;i++){float up=(1.-h0)*float(i)/3.;shade=max(shade,smoothstep(0.,.08,gdHeightAt(uv+shift+climb*up,layer,dx,dy)-(h0+up)));}
 }
 shade*=amount;return shift;
}
`;

const ASPHALT = /* glsl */`
uniform float uGdUnpainted, uRoadDamp;
uniform sampler2D uRoadReflection;
float gdFilm;

float gdBand(float x,float c,float w){return 1.-smoothstep(w*.5,w,abs(x-c));}
vec3 gdAsphalt(vec3 albedo){
 vec2 p=vGdPos.xz;vec2 en=vec2(p.x,-p.y);
 float age=gdFbm(p*.06+7.3);                                   // older (greyer, lighter) or newer (darker) surfacing, ~15 m
 float blotch=gdFbm(p*.25+1.7);                                 // stains and wear, ~4 m
 // In the default unpainted scan the stone is ochre. Binder darkens and
 // neutralises it; authored pavement colours and painted lanes keep their hue.
 float y=dot(albedo,vec3(.2126,.7152,.0722));
 vec3 base=mix(albedo,mix(albedo,vec3(y),.75)*vec3(.84,.85,.87),uGdUnpainted);
 // Weathered city asphalt reflects about 12-15 % of the light on it; the scanned texture is fresh binder at about 7 %
 // (measured in the viewer against the white car and the concrete pavement: sunlit road sRGB 68 where street
 // photographs show 120-140). Lift it to the weathered value so the wear, cracks and shade pattern are not crushed.
 base*=2.;
 vec3 c=base*mix(.84,1.18,smoothstep(.3,.7,age))*mix(.91,1.08,blotch);
 // the road field (roadfield.ts): which way the road runs, how far the kerb is, how wide the road is
 vec4 rf=rfAt(vGdPos);float on=smoothstep(.2,.6,rf.y);float d=rf.x;
 vec2 g=rfAcross(vGdPos,d);vec2 t=vec2(-g.y,g.x);if(dot(t,vec2(1.,.37))<0.)t=-t;   // one sense along the road
 float s=dot(en,t),hw=max(rf.y,.5),w=2.*hw;
 float lanes=max(1.,floor(w/3.3+.35)),lw=w/lanes,q=fract(d/lw);
 float laneLike=on*smoothstep(.5,.9,rf.z)*smoothstep(1.2,2.,hw);      // along a street, not in an intersection
 // wheel paths: two polished, rubber-darkened bands per lane, broken up along the road
 float wheel=(gdBand(q,.24,.18)+gdBand(q,.76,.18))*laneLike*mix(.58,1.,gdNoise(p*.48));
 c*=1.-.16*wheel;
 // oil drips down the middle of each lane: dark smudges stretched along it
 float oil=gdBand(q,.5,.11)*laneLike*smoothstep(.55,.85,gdNoise(vec2(s*.45,q*9.+3.)))*smoothstep(.35,.6,gdNoise(vec2(s*.05,7.)));
 c*=1.-.3*oil;
 // the gutter: grit, leaves and run-off darken the last half metre before the kerb
 float gutter=on*(1.-smoothstep(.15,.75,d));c*=1.-.25*gutter*mix(.6,1.,gdNoise(p*1.7));
 // A visible road repair needs an observed footprint. Do not paint large
 // rectangular works into a street automatically from arbitrary grid cells.
 // cracks, sealed with tar: across the road every 6-14 m in older stretches (wandering, not always kerb to kerb); plain
 // wandering cracks where the road is not lane-like
 float older=smoothstep(.42,.62,age);
 float cs=s/9.;float ci=floor(cs);float cpos=.2+.6*gdHash(vec2(ci,4.1));
 float wob=(gdNoise(vec2(d*.35,ci*7.1))-.5)*1.1+(gdNoise(vec2(d*2.3,ci*3.3))-.5)*.18;
 float xd=abs((fract(cs)-cpos)*9.-wob);
 float across=step(gdHash(vec2(ci,9.7)),.6)*smoothstep(.25,.45,gdNoise(vec2(d*.5,ci*1.9)))*laneLike;
 float crack=across*(1.-smoothstep(.006,.02,xd))*older;
 float seal=across*(1.-smoothstep(.03,.07,xd))*older;                    // the tar sealant band either side of the crack
 float wander=(1.-smoothstep(.0,.022,gdEdge(p*.45+vec2(gdNoise(p*.9)*.6))))*smoothstep(.6,.72,gdFbm(p*.035+5.1))*older*(1.-laneLike);
 c*=1.-.35*seal*gdKeep(8.);c*=1.-.55*max(crack,wander)*gdKeep(20.);
 // Exposed chips and darker binder at the centimetre scale. Blend two
 // nearby world samples along traffic flow, so changing street direction
 // never changes the phase of the grain.
 // The old road's cracks and hollows have real depth (3 cm, a little more than life so it reads from a car): relief on
 // the 8 m layer within 13 m of the eye. Everything finer rides on the shifted surface.
 vec2 pw=p*.125;vec3 eyeW=cameraPosition-vGdPos;float eyeD=length(eyeW);
 float reliefOn=(1.-smoothstep(8.,13.,eyeD))*(.35+.65*older)*(1.-smoothstep(.35,.65,gdNoise(pw*.37+18.6)));
 float reliefShade;vec2 lift=gdRelief(pw,dFdx(pw),dFdy(pw),6.,eyeW/eyeD,uGdSun,.03*.125,reliefOn,reliefShade);
 p+=lift*8.;
 float coarse=gdNoise(p*18.);
 float coarseKeep=gdKeep(18.);
 c*=1.+(coarse-.5)*.1*coarseKeep;
 // The surfacing itself, from the generated layers: dense recent asphalt where the road is newer, worn asphalt with
 // pale stones standing out of the binder where it is older, and broken, ravelled patches in some old stretches away
 // from the wheel paths. Traffic polishes and darkens the stones in the wheel paths.
 vec3 surf=mix(gdLayer(p,0.),gdLayer(p,1.),smoothstep(.38,.66,age+.25*(blotch-.5)));
 float broken=smoothstep(.6,.72,gdFbm(p*.11+11.3))*older*(1.-.7*wheel);
 if(broken>.003)surf=mix(surf,gdLayer(p*.8+3.3,2.),broken);
 // a second look at the worn layer four times larger: stone-rich and binder-rich patches that still read at 20-40 m
 vec2 pm=p*.23;float patchy=gdLayerAt(pm,1.,dFdx(pm),dFdy(pm)).z;
 // old surfacing at the scale of a lane: sealed and open cracks, mottling and stains (an 8 m tile, two copies traded)
 vec3 wide=gdLayer(pw,6.,lift);float aged=.35+.65*older;
 c*=(.5+1.*surf.z)*(.86+.28*patchy)*(1.-.08*wheel)*(1.+1.8*(wide.z-.5)*aged);
 gdSlope=surf.xy*(1.-.45*wheel)+wide.xy*aged;
 c*=1.-.5*reliefShade;
 gdHeight=(coarse-.5)*.3*coarseKeep-max(crack,wander)*.25*gdKeep(20.)-seal*.05*gdKeep(8.);
 // Add the wear contribution to the scanned binder roughness below. Polished
 // tracks and tar reflect a lamp more tightly than the weathered shoulders.
 // Dry asphalt has a broad, diffuse microfacet lobe. Preserve a small polish
 // in wheel tracks and tar, but avoid a continuous sun-facing sheen.
 gdRough=.62+.08*age-.16*wheel-.14*seal+.07*gutter+.04*(coarse-.5)*coarseKeep;
 gdRough+=(.5-surf.z)*.22;                                          // stone faces are smoother than the binder between them
 gdFilm=0.;
 if(uRoadDamp>.5){
  // Shallow water hangs on in depressions and beside the crown of the road.
  // Filter out standing water on steep streets; a thin film can remain there.
  vec3 plane=cross(dFdx(vGdPos),dFdy(vGdPos));
  float level=abs(plane.y)/max(.000001,length(plane));
  float hollow=blotch+.12*gutter+.045*wheel-.035*coarse;
  gdFilm=.16+.84*smoothstep(.49,.69,hollow)*smoothstep(.984,.999,level);
  c*=1.-.44*gdFilm;
  gdHeight*=1.-.75*gdFilm;
 }
 return c;
}
`;

// The coverage mask stays inside the actual stripe or glyph geometry.
const ROAD_PAINT = /* glsl */`
vec3 gdRoadPaint(vec3 pigment){
 vec2 p=vGdPos.xz;
 float age=smoothstep(.33,.73,gdNoise(p*.59));
 float flecks=gdNoise(p*39.);
 float worn=smoothstep(.64,.84,flecks)*age;
 // Once individual bare spots are subpixel, keep their mean coverage.
 float chips=mix(.040*age,worn,gdKeep(39.));
 float grit=(gdNoise(p*13.)-.5)*gdKeep(13.);
 vec3 paint=pigment*(.90+.09*(1.-age)+.12*grit);
 gdRough=clamp(.46+.18*age+.18*chips,.43,.85);
 gdHeight=grit*.22+(flecks-.5)*.28*gdKeep(39.)-chips*.33;
 return mix(paint,vec3(.085,.084,.080),chips*.84);
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
 // near the eye the blades stand on this surface (grass-blades.ts): what shows between them is their shaded roots
 c*=mix(.5,1.,smoothstep(8.,34.,distance(cameraPosition,vGdPos)));
 return c;
}
`;

const GARDEN = /* glsl */`
vec3 gdGarden(vec3 albedo){
 vec2 p=vGdPos.xz;
 // what the bare terrain shows is mostly back gardens: lawn, darker planting beds and bare soil
 float kind=gdFbm(p*.11+3.3),wear=gdFbm(p*.6+8.1),bed=gdFbm(p*.22+6.6);
 vec3 lawn=vec3(.15,.21,.09)*mix(.7,1.12,gdFbm(p*.35+1.)),planting=vec3(.07,.11,.05)*mix(.8,1.2,wear),soil=vec3(.25,.19,.13)*mix(.8,1.15,wear);
 vec3 c=mix(lawn,planting,smoothstep(.56,.62,bed));c=mix(c,soil,(1.-smoothstep(.27,.3,kind)));
 // Patios/decks are mapped surfaces. Unknown yard ground has no synthetic
 // pavement rectangles that could be mistaken for real hardscape.
 // where the terrain shows in the narrow gaps between a road and its kerb, or along a building's foot, it is the
 // gutter's or the wall foot's grime, not lawn (roadfield.ts: distance from the road's edge and from the nearest wall)
 vec4 rf=rfAt(vGdPos);float gap=max(1.-smoothstep(.45,.9,abs(rf.x)),1.-smoothstep(.5,1.,rf.w));
 c=mix(c,albedo*vec3(.42,.41,.39)*mix(.8,1.1,wear),gap);
 c*=mix(1.,.86+.28*gdNoise(p*12.),gdKeep(12.));
 gdHeight=1.*gdNoise(p*6.)*gdKeep(6.);
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
 // The slab's own surface, from the generated layers: most are broomed (ridges across the walk), the older ones have
 // weathered down to their sand and stones, a recent pour is trowelled smooth. Each slab takes its own piece of the
 // tile, so no two neighbours match; the kerb's face is wrapped along its height.
 vec2 across=vec2(-a.y,a.x);float onTop=step(.5,level);
 vec2 dpx=dFdx(p),dpy=dFdy(p);
 vec2 luv=mix(vec2(dot(p,a),vGdPos.y),vec2(dot(p,across),dot(p,a)),onTop)*.7;
 vec2 ldx=mix(vec2(dot(dpx,a),dFdx(vGdPos.y)),vec2(dot(dpx,across),dot(dpx,a)),onTop)*.7,ldy=mix(vec2(dot(dpy,a),dFdy(vGdPos.y)),vec2(dot(dpy,across),dot(dpy,a)),onTop)*.7;
 vec3 surf=gdLayerAt(luv+vec2(r3,r4)*7.,fresh?5.:(r1<.58?3.:4.),ldx,ldy);
 // the walk as a whole: stains, run-off and long cracks that ignore the joints (an 8 m tile)
 vec2 wuv=vec2(dot(p,across),dot(p,a))*.125;vec3 wide=gdLayerAt(wuv,7.,vec2(dot(dpx,across),dot(dpx,a))*.125,vec2(dot(dpy,across),dot(dpy,a))*.125);
 float old=fresh?.25:1.;
 c*=(.56+.88*surf.z)*(1.+2.*(wide.z-.5)*old*onTop);
 gdSlope=(across*(surf.x+wide.x*old)+a*(surf.y+wide.y*old))*level;
 // a broom finish: fine ridges along the joints, seen only close to
 float broom=gdNoise(vec2(t*140.,fb*3.+r1*9.));
 c*=mix(1.,.97+.06*broom,gdKeep(140.)*level);
 c*=1.-level*(.16*halo+.5*joint+.55*crack);
 // The bevel is wider than the scored groove. Its sloped sides catch
 // light from opposite directions as the street lights move past the driver.
 float groove=min(jx,jy);
 float bevel=(1.-smoothstep(.004,.018,groove))*gdKeep(22.)*level;
 float grit=(gdNoise(p*42.)-.5)*gdKeep(42.);
 c*=1.+.19*grit;
 gdHeight=gdNoise(p*7.)*gdKeep(7.)*.32+broom*.15*gdKeep(140.)+grit*.25-1.4*bevel-.45*crack;
 gdRough=1.-.13*walk+.05*spill+.12*bevel+.10*grit+(fresh?.04:0.)+(.5-surf.z)*.2;   // a factor on the scan's own roughness
 return c;
}
`;

function patch(materials: Iterable<THREE.MeshStandardMaterial>, tag: string, body: string, call: string, bump: number, roughness: string, uniforms: Record<string, THREE.IUniform> = {}, extra = '') {
  for (const m of materials) {
    if (m.userData.groundDetail) continue; m.userData.groundDetail = tag;
    // Road and paving are not metal: the metalness map only repeats the roughness texture on a second texture unit, and
    // the layered ground materials need that unit (a fragment shader may use 16).
    if ('uGdLayers' in uniforms && m.metalnessMap) { m.metalnessMap = null; m.metalness = 0; }
    const previous = m.onBeforeCompile.bind(m);
    m.onBeforeCompile = (shader, renderer) => {
      previous(shader, renderer); Object.assign(shader.uniforms, uniforms);
      shader.vertexShader = shader.vertexShader.replace('#include <common>', '#include <common>\nvarying vec3 vGdPos;')
        .replace('#include <project_vertex>', '#include <project_vertex>\nvGdPos=(modelMatrix*vec4(transformed,1.)).xyz;');
      shader.fragmentShader = shader.fragmentShader.replace('#include <common>', '#include <common>\n' + COMMON + extra + body)
        .replace('#include <map_fragment>', `#include <map_fragment>\ngdSlope=vec2(0.);diffuseColor.rgb=${call}(diffuseColor.rgb);`)
        .replace('#include <roughnessmap_fragment>', `#include <roughnessmap_fragment>\nroughnessFactor=${roughness};`)
        .replace('#include <normal_fragment_maps>', `#include <normal_fragment_maps>\nnormal=gdBump(normal,gdHeight,${bump.toFixed(4)});normal=normalize(normal+mat3(viewMatrix)*vec3(gdSlope.x,0.,gdSlope.y));`);
    };
    const key = m.customProgramCacheKey.bind(m);
    m.customProgramCacheKey = () => key() + '|ground-' + tag;
    m.needsUpdate = true;
  }
}

/** Wear, lane wheel paths, oil, gutters, patches, sealed cracks and aggregate on road asphalt. `field` holds the road
 *  field's uniforms (roadfield.ts); its texture may be set after this call, before the first frame. */
export function detailAsphalt(materials: Iterable<THREE.MeshStandardMaterial>, field: Record<string, THREE.IUniform>) {
  for (const material of materials) {
    loadLayers();
    const uniforms = {...field, uGdLayers: layers, uGdSun: groundSun, uGdUnpainted: {value: material.name === 'asphalt' ? 1 : 0}};
    patch([material], 'asphalt5', ASPHALT, 'gdAsphalt', .0035,
        'mix(clamp(gdRough + .40 * roughnessFactor, .64, .98),mix(.24,.13,gdFilm),gdFilm)', uniforms, ROADFIELD_GLSL + LAYERS);
    // Sample the completed street probe. The road keeps its calibrated diffuse
    // lighting; the water film reflects the neighbourhood at grazing angles.
    const prior=material.onBeforeCompile.bind(material);
    material.onBeforeCompile=(shader,renderer)=>{
      prior(shader,renderer);
      shader.fragmentShader=shader.fragmentShader.replace('#include <aomap_fragment>', `
        #include <aomap_fragment>
        #ifdef ENVMAP_TYPE_CUBE_UV
        if(gdFilm>.20){
          vec3 eye=normalize(vViewPosition);
          vec3 R=inverseTransformDirection(reflect(-eye,normal),viewMatrix);
          #ifdef STREET_DEPTH_PROBE
          R=correctStreetDirection(uRoadReflection,vGdPos,R,mix(.24,.13,gdFilm));
          #endif
          float facing=clamp(dot(eye,normal),0.,1.);
          float fresnel=.02+.98*pow(1.-facing,5.);
          // PMREM's first three levels are nested at the same edge of its atlas.
          vec2 atlas=vec2(textureSize(uRoadReflection,0));
          float largest=atlas.y*.25;
          float faceSize=largest*exp2(-floor(mix(2.,1.,gdFilm)));
          float face=getFace(R);
          vec2 uv=getUV(R,face)*(faceSize-2.)+1.;
          if(face>2.){uv.y+=faceSize;face-=3.;}
          uv+=vec2(face*faceSize,4.*(largest-faceSize));
          vec3 street=textureLod(uRoadReflection,uv/atlas,0.).rgb;
          reflectedLight.indirectSpecular=mix(reflectedLight.indirectSpecular,street*fresnel,gdFilm*.9);
        }
        #endif`);
    };
    const key=material.customProgramCacheKey.bind(material);
    material.customProgramCacheKey=()=>key()+'|water-film';
  }
}

/** Put a wear finish on flat road-marking pigment. A shared white paint may
 * also be used by building fascia, kerbs or sign faces elsewhere in the GLB. */
export function detailRoadPaint(meshes: THREE.Mesh[],
  originals: Map<THREE.Mesh, THREE.Material | THREE.Material[]>,
  setup: (material: THREE.Material) => void) {
  const copies = new Map<THREE.MeshStandardMaterial, THREE.MeshStandardMaterial>();
  function roadFinish(material: THREE.Material) {
    if (!(material instanceof THREE.MeshStandardMaterial) || material.map || material.vertexColors) return material;
    let copy = copies.get(material);
    if (!copy) {
      copy = material.clone();
      setup(copy);
      copies.set(material, copy);
    }
    return copy;
  }
  for (const mesh of meshes) {
    mesh.material = Array.isArray(mesh.material) ? mesh.material.map(roadFinish) : roadFinish(mesh.material);
    originals.set(mesh, mesh.material);
  }
  patch(copies.values(), 'road-paint', ROAD_PAINT, 'gdRoadPaint', .003, 'gdRough');
}

/** Lush and dry patches, mowing stripes, blade grain and relief on lawns. */
export function detailGrass(materials: Iterable<THREE.MeshStandardMaterial>) {
  patch(materials, 'grass2', GRASS, 'gdGrass', .012, 'clamp(max(roughnessFactor,.9),.0,1.)');
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
  loadLayers();
  patch(materials, 'pavement5', PAVEMENT, 'gdPavement', .004, 'clamp(roughnessFactor*gdRough,.45,1.)', {...fields, uGdLayers: layers, uGdSun: groundSun}, ROADFIELD_GLSL + PAVING_GLSL + LAYERS);
}

/** Ground classes this module details, by a mesh's material class. */
export const ASPHALT_CLASS = /^asphalt/, GRASS_CLASS = /^(grass|turf|lawn)/;
