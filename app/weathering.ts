import * as THREE from 'three';
import {FACADEMAP_GLSL} from './facademap';

// Weathering: what makes a wall read as built and lived with rather than printed. The generated world gives every
// facade one material, one colour and a texture that repeats every few metres; real walls are dirtier at the foot
// (splash-back and street grime within a metre and a half of the pavement), carry faint rain streaks, and drift in
// tone over tens of metres; flat roofs are patched membranes. All of it is procedural in world space, measured against
// the terrain height under each point, so it follows every building without any per-building data. Glazing, signs and
// the ground are left alone.

const HEADER = /* glsl */`
uniform sampler2D uWxGround;uniform vec4 uWxGrid;uniform sampler2D uWxHeight;uniform vec2 uWxH;
varying vec3 vWxPos;varying vec3 vWxNrm;
float wxLarge;float wxDirt;
float wxHash(vec2 p){return fract(sin(dot(p,vec2(127.1,311.7)))*43758.5453);}
float wxNoise(vec2 p){vec2 i=floor(p),f=fract(p);f=f*f*(3.-2.*f);return mix(mix(wxHash(i),wxHash(i+vec2(1.,0.)),f.x),mix(wxHash(i+vec2(0.,1.)),wxHash(i+vec2(1.,1.)),f.x),f.y);}
float wxFbm(vec2 p){float v=0.,a=.5;for(int i=0;i<4;i++){v+=a*wxNoise(p);p=p*2.03+vec2(17.,9.);a*=.5;}return v;}
vec3 wxTint(vec3 albedo){
 vec3 P=vWxPos;vec3 N=normalize(vWxNrm);
 vec2 g=((vec2(P.x,-P.z)-uWxGrid.xx)/uWxGrid.y+.5)/uWxGrid.z;float h=P.y-texture2D(uWxGround,g).r;
 float wall=smoothstep(.3,.7,1.-abs(N.y));
 vec3 T=normalize(vec3(-N.z,0.,N.x)+vec3(1e-5,0.,0.));float u=dot(P.xz,T.xz);
 wxLarge=wxFbm(vec2(u,P.y)*.045+P.xz*.013);
 // the roof line over this point of the wall: the height map just inside the wall (sky visibility's, occlusion.ts)
 float below=8.;
 if(uWxH.y>.5){vec2 xz=P.xz-normalize(N.xz+vec2(1e-5))*.9;vec2 uv=vec2(xz.x,-xz.y)/(2.*uWxH.x)+.5;
  if(uv.x>0.&&uv.y>0.&&uv.x<1.&&uv.y<1.)below=max(texture2D(uWxHeight,uv).r-P.y,0.);}
 // run-off: dirt washed off the roof hangs from the roof line in columns of uneven width and length
 // thin streaks (about 10-40 cm) gathered in uneven groups, each breaking up and fading at its own length
 float group=smoothstep(.35,.75,wxNoise(vec2(u*.35,7.3))*.7+wxNoise(vec2(u*1.1,1.9))*.3);
 float thin=smoothstep(.55,.9,wxNoise(vec2(u*4.2,.5))*.6+wxNoise(vec2(u*11.,2.1))*.4);
 float len=mix(1.2,8.,wxNoise(vec2(u*2.3,11.)));
 float broken=smoothstep(.25,.7,wxNoise(vec2(u*6.,P.y*.8))*.7+wxNoise(vec2(u*17.,P.y*2.1))*.3);
 float runoff=group*thin*broken*exp(-below/len);
 // and a broad, faint wash over the top storeys of a wall
 float wash=exp(-below/16.);
 // splash-back and street grime at the foot, with an uneven top edge
 float grime=1.-smoothstep(0.,1.3+1.3*wxNoise(vec2(u*.6,1.)),h);
 // tone mottling at a metre and a few metres, which breaks a repeating texture's grid
 float mottle=wxNoise(vec2(u,P.y)*1.7)*.5+wxNoise(vec2(u,P.y)*.55+3.)*.5;
 vec3 t=vec3(mix(.91,1.07,wxLarge))*mix(vec3(.985,1.,1.025),vec3(1.025,1.,.97),wxLarge);
 t*=mix(1.,mix(.95,1.05,mottle),wall);
 wxDirt=wall*clamp(runoff*.75+wash*.16,0.,.65);
 t*=1.-wxDirt*.42;
 t*=1.-wall*.24*grime;
 // flat roofs: patched membranes, and over tens of metres a mix of dark tar-and-gravel and light membranes
 float patches=wxFbm(P.xz*.08);
 float roofTone=smoothstep(.4,.8,wxFbm(P.xz*.012+3.1));
 t=mix(t*vec3(mix(.84,1.1,patches))*mix(.85,2.4,roofTone),t,wall);
 vec3 c=albedo*t;float l=dot(c,vec3(.2126,.7152,.0722));
 // dirt and grime are a grey-brown that takes the colour out of what they cover
 return mix(c,vec3(l)*vec3(1.04,1.,.93),clamp(.28*grime*wall+wxDirt*.55,0.,.6));
}

// Surface character by kind (uWxKind: 0 none, 1 render/stucco, 2 concrete, 3 cladding panels, 4 dressed stone): what a
// real wall of that kind shows at a few metres that its texture does not. All in wall-local metres (u along the wall, y
// up). Around its windows (facademap.ts) a wall of render, concrete or cladding carries the dirt washed off every sill
// down the wall below it, mostly from the sill's two ends, and render cracks from the corners of its openings.
uniform float uWxKind;
float wxH;
float wxKeep(float x,float f){return 1.-smoothstep(.25,.5,fwidth(x)*f);}
// the share of a pixel px metres wide covered by a hairline of half width w at signed distance o
float wxLine(float o,float w,float px){return max(0.,min(o+.5*px,w)-max(o-.5*px,-w))/px;}
// a crack running from a corner (q: the point less the corner, wall metres) along dir for up to len metres, wandering
float wxCrackFrom(vec2 q,vec2 dir,float len,float seed,float px){
 float s=dot(q,dir);if(s<0.||s>len)return 0.;
 float o=dot(q,vec2(-dir.y,dir.x))-(wxNoise(vec2(s*5.,seed))-.5)*.14-(wxNoise(vec2(s*21.,seed+3.))-.5)*.035;
 return wxLine(o,.0013,px)*(1.-smoothstep(.55*len,len,s));
}
vec3 wxSurface(vec3 albedo){
 vec3 P=vWxPos;vec3 N=normalize(vWxNrm);wxH=0.;float px=max(length(fwidth(P))*.7,1e-4);
 float wall=smoothstep(.3,.7,1.-abs(N.y));if(wall<.01||uWxKind<.5||uWxKind>3.5)return albedo;
 vec3 T=normalize(vec3(-N.z,0.,N.x)+vec3(1e-5,0.,0.));float u=dot(P.xz,T.xz),y=P.y;
 // each wall plane a slightly different tone: buildings of one material are painted, poured and weathered apart
 float plane=floor(dot(P.xz,N.xz)*2.+.5);vec2 nq=floor(N.xz*8.+.5);
 float wt=wxHash(nq*17.3+vec2(plane*.37,plane*.11));
 vec3 c=albedo*(.95+.1*wt);
 if(uWxKind<1.5){
  // render: paint fading unevenly (0.3-1 m), now and then a repaired patch, a fine sand-float grain
  c*=.94+.12*wxFbm(vec2(u,y)*1.3+7.7);
  c*=.9+.2*smoothstep(.25,.75,wxFbm(vec2(u,y)*.22+3.3));                         // repainted bays and fading over metres
  vec2 cell=floor(vec2(u,y)/vec2(2.3,1.7));float pick=wxHash(cell+nq*3.1);
  if(pick<.08){vec2 q=fract(vec2(u,y)/vec2(2.3,1.7));vec2 lo=wxHash(cell+4.4)*vec2(.4,.4),hi=lo+vec2(.3)+wxHash(cell+8.8)*vec2(.3);
   float inside=step(lo.x,q.x)*step(q.x,hi.x)*step(lo.y,q.y)*step(q.y,hi.y);c*=mix(1.,.9+.18*wxHash(cell+2.2),inside);}
  float grain=wxNoise(vec2(u,y)*190.)*.6+wxNoise(vec2(u,y)*71.)*.4;
  c*=mix(1.,.95+.1*grain,wxKeep(u,190.));wxH=grain*wxKeep(u,190.);
 }else if(uWxKind<2.5){
  // concrete: formwork panels (2.4 x 1.2 m) with their seams, four tie holes each, each pour a different tone
  vec2 q=vec2(u,y)/vec2(2.4,1.2);vec2 pc=floor(q);vec2 f=fract(q);
  c*=.93+.14*wxHash(pc+nq*5.3);
  vec2 e=min(f,1.-f)*vec2(2.4,1.2);float seam=1.-smoothstep(.004,.014,min(e.x,e.y));
  c*=1.-.22*seam*wxKeep(u,40.);
  vec2 th=fract(f*vec2(2.,2.))-.5;float hole=1.-smoothstep(.012,.02,length(th*vec2(1.2,.6)));
  c*=1.-.45*hole*wxKeep(u,30.);
  // lime bleeding from the seams, pale streaks below the horizontal ones
  float bleed=smoothstep(.0,.25,1.-f.y)*(1.-smoothstep(.25,.6,1.-f.y))*smoothstep(.55,.85,wxNoise(vec2(u*3.,pc.y)));
  c=mix(c,c*1.12+.015,.5*bleed);
  float grain=wxNoise(vec2(u,y)*120.)*.6+wxNoise(vec2(u,y)*43.)*.4;c*=mix(1.,.94+.12*grain,wxKeep(u,120.));wxH=grain*wxKeep(u,120.)-seam*.6;
 }else{
  // cladding panels: joints on a 1.2 x 3 m module, each panel a little different, a faint oil-canning shimmer
  vec2 q=vec2(u,y)/vec2(1.2,3.);vec2 pc=floor(q);vec2 f=fract(q);
  c*=.95+.1*wxHash(pc+nq*7.1);
  vec2 e=min(f,1.-f)*vec2(1.2,3.);float joint=1.-smoothstep(.006,.016,min(e.x,e.y));
  c*=1.-.5*joint*wxKeep(u,40.);wxH=-joint+.3*wxNoise(f*vec2(2.,5.)+pc);
 }
 // the windows in front of whose columns this point stands
 float streak=0.,crack=0.;ivec2 ent=wmEntries(P);
 for(int i=0;i<48;i++){
  if(i>=ent.y)break;
  int ri=wmListAt(ent.x+i);vec4 A=wmRunA(ri),B=wmRunB(ri);vec2 ax=A.xy;
  if(abs(dot(N.xz,ax))>.3||abs(dot(P.xz,vec2(-ax.y,ax.x))-A.z)>.8)continue;      // a wall of the window's plane
  float uu=dot(P.xz,ax);if(uu<B.x-.9||uu>B.y+.9)continue;
  float cnt=floor(B.w),hh=fract(B.w)*10.,pitch=B.z,y0=A.w;
  float k=pitch>.01?max(ceil((P.y-y0)/pitch),0.):P.y<=y0?0.:1.;                  // the window above: the first sill at or over P
  float hs=fract(sin(float(ri)*12.9898+k*4.1)*43758.5453);
  if(k<cnt){
   float d=y0+k*pitch-P.y;                                                     // metres under that sill
   float fade=exp(-d/mix(.6,2.8,hs*hs));
   float w0=.07+.04*d;
   float ends=exp(-pow((uu-B.x-.05)/w0,2.))+exp(-pow((uu-B.y+.05)/w0,2.));
   float inside=smoothstep(B.x-.02,B.x+.06,uu)*smoothstep(B.y+.02,B.y-.06,uu);
   float thin=inside*smoothstep(.5,.85,wxNoise(vec2(uu*7.+float(ri)*3.1,d*.35+hs*9.)));
   streak=max(streak,(.85*ends+.55*thin)*fade*mix(.45,1.,hs)+.3*inside*exp(-d/.35));
   // render cracks down and out from the sill's corners
   if(uWxKind<1.5){vec2 q=vec2(uu,P.y)-vec2(B.x,y0+k*pitch);
    if(hs>.6)crack=max(crack,wxCrackFrom(q,normalize(vec2(-.55,-1.)),.5+1.1*hs,hs*31.,px));
    q.x=uu-B.y;if(fract(hs*7.3)>.6)crack=max(crack,wxCrackFrom(q,normalize(vec2(.55,-1.)),.5+1.1*fract(hs*3.7),hs*17.,px));}
  }
  // and up and out from the head corners of the window below
  if(uWxKind<1.5&&k>.5&&k-1.<cnt){float yh=y0+(k-1.)*pitch+hh,hb=fract(sin(float(ri)*7.13+k*2.9)*43758.5453);
   vec2 q=vec2(uu-B.x,P.y-yh);if(hb>.65)crack=max(crack,wxCrackFrom(q,normalize(vec2(-.5,1.)),.4+.9*hb,hb*23.,px));
   q.x=uu-B.y;if(fract(hb*5.1)>.65)crack=max(crack,wxCrackFrom(q,normalize(vec2(.5,1.)),.4+.9*fract(hb*2.3),hb*41.,px));}
 }
 // dirt off the sills is a grey-brown: it darkens a pale wall and pales a dark one
 c=mix(c,vec3(.17,.155,.135),.48*clamp(streak,0.,1.));
 c*=1.-.45*crack;
 return mix(albedo,c,wall);
}
vec3 wxBump(vec3 n,float h,float k){vec3 dpx=dFdx(-vViewPosition),dpy=dFdy(-vViewPosition);float hx=dFdx(h),hy=dFdy(h);
 vec3 r1=cross(dpy,n),r2=cross(n,dpx);float det=dot(dpx,r1);vec3 g=sign(det)*(hx*r1+hy*r2);return normalize(abs(det)*n-k*g);}
`;

// Layers that are weathered, and the surface classes on them that are not (glazing, interiors behind it, metal fixings).
export const WEATHERED_LAYERS = new Set(['buildings', 'facades', 'roofs', 'foundations']);
export const UNWEATHERED = /^(glass|curtain|interior|screen|bottle|water|lamp|steel|aluminum|silver|zinc|copper|chrome)/;

export function groundTexture(grid: { min: number; step: number; n: number }, ground: ArrayLike<number>) {
  const tex = new THREE.DataTexture(Float32Array.from(ground), grid.n, grid.n, THREE.RedFormat, THREE.FloatType);
  tex.magFilter = tex.minFilter = THREE.LinearFilter; tex.wrapS = tex.wrapT = THREE.ClampToEdgeWrapping; tex.needsUpdate = true;
  return tex;
}

/** Adds weathering to the given materials; chained after any existing onBeforeCompile (the shadow cascades'). */
// the surface character of a wall material class (wxSurface): render and stucco, concrete, cladding panels; brick, stone,
// timber and roofs keep their own textures
export function surfaceKind(cls: string) { return /^(render|stucco)/.test(cls) ? 1 : /^(concrete|venue_concrete|cast_stone)/.test(cls) ? 2 : /^(panel_|terracotta_panel|curtain_panel)/.test(cls) ? 3 : /^(stone|limestone)/.test(cls) ? 4 : 0; }
export function weatherMaterials(materials: Iterable<THREE.MeshStandardMaterial>, tex: THREE.DataTexture, grid: { min: number; step: number; n: number }, kinds?: Map<THREE.Material, number>) {
  // the facade map's tables (facademap.ts) are set once the windows are known: setFacadeMap below
  const uniforms = { uWxGround: { value: tex }, uWxGrid: { value: new THREE.Vector4(grid.min, grid.step, grid.n, 0) }, uWxHeight: { value: null as THREE.Texture | null }, uWxH: { value: new THREE.Vector2(1, 0) },
    uWmCol: { value: null as THREE.Texture | null }, uWmCell: { value: null as THREE.Texture | null }, uWmList: { value: null as THREE.Texture | null }, uWmGrid: { value: new THREE.Vector4() } };
  for (const m of materials) {
    if (m.userData.weathered) continue; m.userData.weathered = true;
    const kind = kinds?.get(m) ?? 0;
    const previous = m.onBeforeCompile.bind(m);
    m.onBeforeCompile = (shader, renderer) => {
      previous(shader, renderer);
      Object.assign(shader.uniforms, uniforms); shader.uniforms.uWxKind = {value: kind};
      shader.vertexShader = shader.vertexShader.replace('#include <common>', '#include <common>\nvarying vec3 vWxPos;varying vec3 vWxNrm;')
        .replace('#include <project_vertex>', '#include <project_vertex>\nvWxPos=(modelMatrix*vec4(transformed,1.)).xyz;vWxNrm=normalize(mat3(modelMatrix)*objectNormal);');
      shader.fragmentShader = shader.fragmentShader.replace('#include <common>', '#include <common>\n' + FACADEMAP_GLSL + HEADER)
        // dressed stone: the scan's blotches (a metre across) kept at a quarter, its grain whole
        .replace('#include <map_fragment>', '#include <map_fragment>\n#ifdef USE_MAP\nif(uWxKind>3.5){vec3 t0=texture2D(map,vMapUv).rgb,m1=textureLod(map,vMapUv,4.).rgb,m2=textureLod(map,vMapUv,10.).rgb;diffuseColor.rgb*=max(m2+(t0-m1)+.25*(m1-m2),vec3(0.))/max(t0,vec3(1e-3));}\n#endif\ndiffuseColor.rgb=wxTint(wxSurface(diffuseColor.rgb));')
        .replace('#include <normal_fragment_maps>', '#include <normal_fragment_maps>\nif(uWxKind>.5)normal=wxBump(normal,wxH,.0025);')
        .replace('#include <roughnessmap_fragment>', '#include <roughnessmap_fragment>\nroughnessFactor=clamp(roughnessFactor*mix(.9,1.08,wxLarge)*(1.+.15*wxDirt),.04,1.);');
    };
    const key = m.customProgramCacheKey.bind(m);
    m.customProgramCacheKey = () => key() + '|weathered2-' + kind;
    m.needsUpdate = true;
  }
  /** The static world's top-down height map (occlusion.ts), for the roof line over each point of a wall. */
  return { setHeight(texture: THREE.Texture, radius: number) { uniforms.uWxHeight.value = texture; uniforms.uWxH.value.set(radius, 1); },
    /** The windows' facade map (facademap.ts), for the streaks, cracks and repainting around them. */
    setFacadeMap(map: {textures: {uWmCol: THREE.Texture; uWmCell: THREE.Texture; uWmList: THREE.Texture}; grid: THREE.Vector4}) { uniforms.uWmCol.value = map.textures.uWmCol; uniforms.uWmCell.value = map.textures.uWmCell; uniforms.uWmList.value = map.textures.uWmList; uniforms.uWmGrid.value.copy(map.grid); } };
}
