import * as THREE from 'three';
import {FACADEMAP_GLSL} from './facademap.ts';

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
// Facet normals are intentionally hard on a curved bay. Weathering must not
// derive its coordinate frame from them: even a small normal change at city
// coordinates shifts a tangent projection by metres and restarts the pattern.
float wxHash3(vec3 p){return fract(sin(dot(p,vec3(127.1,311.7,74.7)))*43758.5453);}
float wxNoise3(vec3 p){
 vec3 i=floor(p),f=fract(p);f=f*f*(3.-2.*f);
 float a=wxHash3(i),b=wxHash3(i+vec3(1.,0.,0.)),c=wxHash3(i+vec3(0.,1.,0.)),d=wxHash3(i+vec3(1.,1.,0.));
 float e=wxHash3(i+vec3(0.,0.,1.)),f0=wxHash3(i+vec3(1.,0.,1.)),g=wxHash3(i+vec3(0.,1.,1.)),h=wxHash3(i+vec3(1.,1.,1.));
 return mix(mix(mix(a,b,f.x),mix(c,d,f.x),f.y),mix(mix(e,f0,f.x),mix(g,h,f.x),f.y),f.z);
}
float wxFbm3(vec3 p){float v=0.,a=.5;for(int i=0;i<4;i++){v+=a*wxNoise3(p);p=p*2.03+vec3(17.,9.,13.);a*=.5;}return v;}
vec3 wxTint(vec3 albedo){
 vec3 P=vWxPos;vec3 N=normalize(vWxNrm);
 vec2 g=((vec2(P.x,-P.z)-uWxGrid.xx)/uWxGrid.y+.5)/uWxGrid.z;float h=P.y-texture2D(uWxGround,g).r;
 float wall=smoothstep(.3,.7,1.-abs(N.y));
 // Weathering is an overlay, separate from the material's authored metric UV
 // texture. A fixed 3D world field stays continuous across curved facets.
 wxLarge=wxFbm3(P*.045);
 // the roof line over this point of the wall: the height map just inside the wall (sky visibility's, occlusion.ts)
 float below=8.;
 if(uWxH.y>.5){vec2 xz=P.xz-normalize(N.xz+vec2(1e-5))*.9;vec2 uv=vec2(xz.x,-xz.y)/(2.*uWxH.x)+.5;
  if(uv.x>0.&&uv.y>0.&&uv.x<1.&&uv.y<1.)below=max(textureLod(uWxHeight,uv,0.).r-P.y,0.);}
 // run-off: dirt washed off the roof hangs from the roof line in columns of uneven width and length
 // thin streaks (about 10-40 cm) gathered in uneven groups, each breaking up and fading at its own length
 float group=smoothstep(.35,.75,wxNoise(P.xz*.35+vec2(7.3,1.9))*.7+wxNoise(P.xz*1.1+vec2(1.9,7.3))*.3);
 float thin=smoothstep(.55,.9,wxNoise(P.xz*4.2+vec2(.5,2.1))*.6+wxNoise(P.xz*11.+vec2(2.1,.5))*.4);
 float len=mix(1.2,8.,wxNoise(P.xz*2.3+vec2(11.,4.)));
 float broken=smoothstep(.25,.7,wxNoise3(P*vec3(6.,.8,6.))*.7+wxNoise3(P*vec3(17.,2.1,17.))*.3);
 float runoff=group*thin*broken*exp(-below/len);
 // and a broad, faint wash over the top storeys of a wall
 float wash=exp(-below/16.);
 // splash-back and street grime at the foot, with an uneven top edge
 float grime=1.-smoothstep(0.,1.3+1.3*wxNoise(P.xz*.6+vec2(1.,3.)),h);
 // tone mottling at a metre and a few metres, which breaks a repeating texture's grid
 float mottle=wxNoise3(P*1.7)*.5+wxNoise3(P*.55+vec3(3.))*.5;
 vec3 t=vec3(mix(.91,1.07,wxLarge))*mix(vec3(.985,1.,1.025),vec3(1.025,1.,.97),wxLarge);
 t*=mix(1.,mix(.95,1.05,mottle),wall);
 wxDirt=wall*clamp(runoff*.75+wash*.16,0.,.65);
 t*=1.-wxDirt*.42;
 t*=1.-wall*.24*grime;
 // Horizontal orientation does not identify a roof material: it also includes
 // cornice tops, sills, steps and painted caps. Preserve the authored albedo;
 // weathering must not randomly convert a dark membrane into a white roof.
 float patches=wxFbm(P.xz*.08);
 t=mix(t*vec3(mix(.97,1.03,patches)),t,wall);
 vec3 c=albedo*t;float l=dot(c,vec3(.2126,.7152,.0722));
 // dirt and grime are a grey-brown that takes the colour out of what they cover
 // These streak positions are an unobserved prior, not measured facade wear.
 // Keep them subtle; the source-selected material owns colour and any actual
 // staining in its maps. Strong procedural washes previously repainted clean
 // pale loggias as brown/grey cavities and changed every observed wall colour.
 vec3 worn=mix(c,vec3(l)*vec3(1.04,1.,.93),clamp(.28*grime*wall+wxDirt*.55,0.,.6));
 return mix(albedo,worn,.2);
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
 // The wall normal decides whether this overlay applies, but never its
 // coordinates. A tangent frame derived from each curved-shell facet changes
 // abruptly at facet boundaries and restarts the apparent finish.
 float wt=wxFbm3(P*.11);
 vec3 c=albedo*(.95+.1*wt);
 // Render: broad paint drift and fine sand-float grain in a fixed world
 // field. Avoid synthetic rectangular repaint patches without source evidence.
 c*=.94+.12*wxFbm3(P*1.3+vec3(7.7));
 c*=.9+.2*smoothstep(.25,.75,wxFbm3(P*.22+vec3(3.3)));
 float grain=wxNoise3(P*190.)*.6+wxNoise3(P*71.)*.4;
 c*=mix(1.,.95+.1*grain,wxKeep(P.x,190.));wxH=grain*wxKeep(P.x,190.);
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
 return mix(albedo,c,.2*wall);
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
// Concrete or cast stone does not prove a 2.4 x 1.2 m formwork grid; a panel
// class does not prove 1.2 x 3 m seams. Structural modules belong to the
// evidence-driven material map. An unknown module keeps its authored surface.
export function surfaceKind(cls: string) { return /^(render|stucco)/.test(cls) ? 1 : 0; }
/** Typed observed patterns already have an authored structural texture and normal map. */
export function hasObservedPattern(meshName: string) {
  const token = meshName.split('___').pop()?.split('#')[0] ?? '';
  return token.includes('~obs1~') && /(?:^|~)pat=[^~]+/.test(token);
}
export function weatherMaterials(materials: Iterable<THREE.MeshStandardMaterial>, tex: THREE.DataTexture, grid: { min: number; step: number; n: number }, kinds?: Map<THREE.Material, number>) {
  // the facade map's tables (facademap.ts) are set once the windows are known: setFacadeMap below
  const uniforms = { uWxGround: { value: tex }, uWxGrid: { value: new THREE.Vector4(grid.min, grid.step, grid.n, 0) }, uWxHeight: { value: null as THREE.Texture | null }, uWxH: { value: new THREE.Vector2(1, 0) },
    uWmCol: { value: null as THREE.Texture | null }, uWmCell: { value: null as THREE.Texture | null }, uWmList: { value: null as THREE.Texture | null }, uWmGrid: { value: new THREE.Vector4() }, uWmCellOffset: { value: new THREE.Vector2() } };
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
        // after the vertex colour (a class's walls share one material and carry their own tint: material-merge.ts)
        .replace('#include <color_fragment>', '#include <color_fragment>\n#ifdef USE_MAP\nif(uWxKind>3.5){vec3 t0=texture2D(map,vMapUv).rgb,m1=textureLod(map,vMapUv,4.).rgb,m2=textureLod(map,vMapUv,10.).rgb;diffuseColor.rgb*=max(m2+(t0-m1)+.25*(m1-m2),vec3(0.))/max(t0,vec3(1e-3));}\n#endif\ndiffuseColor.rgb=wxTint(wxSurface(diffuseColor.rgb));')
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
    setFacadeMap(map: {textures: {uWmCol: THREE.Texture; uWmCell: THREE.Texture; uWmList: THREE.Texture}; grid: THREE.Vector4; cellOffset: THREE.Vector2}) { uniforms.uWmCol.value = map.textures.uWmCol; uniforms.uWmCell.value = map.textures.uWmCell; uniforms.uWmList.value = map.textures.uWmList; uniforms.uWmGrid.value.copy(map.grid); uniforms.uWmCellOffset.value.copy(map.cellOffset); } };
}
