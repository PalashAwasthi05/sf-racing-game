import * as THREE from 'three';

// Surface finish by material class. The class is the last token of a mesh name in the generated world, a join key
// the pipeline writes for every mesh ("glass_dark", "brick_modular#aa4838", "steel"); a colour after '#' never
// changes the finish. Textured classes keep their baked roughness maps and only get a stronger relief and, where a
// real surface of that kind has one, a sheen; flat classes receive the finish of the material they name: glazing
// and metal reflect their surroundings, paint has a satin sheen, masonry and render stay matte. Nothing here
// refers to a place.
type Finish={roughness?:number;metalness?:number;probe?:boolean;env?:number;normal?:number;tint?:[number,number,number]};
// Glazing is a dark, clear dielectric: its reflection is the Fresnel of glass (weak face-on, strong at grazing angles),
// not a mirror. Frames and fixings are satin anodised or painted metal; only bare fixtures read as bright metal.
const GLAZING:Finish={roughness:.06,metalness:0,probe:true,env:1.3};
const METAL:Finish={roughness:.5,metalness:.35,probe:true,env:.5};
// Rough dielectrics (paint, timber, terracotta, dark panels, asphalt) take the scene's sky-and-bounce environment like every
// other surface: a material given the street probe as its envMap takes its diffuse light from it too, scaled by its
// reflection strength, and these read murky in shade (asphalt at a third of the light of the ground beside it, and blue
// from mirroring the sky at a driver's grazing angle). Only glazing, water and metals keep the probe.
const PAINT:Finish={roughness:.72,metalness:0};
const RENDER:Finish={roughness:.85,normal:1.4};
const FINISH:Record<string,Finish>={
 glass:GLAZING,glass_dark:GLAZING,glass_light:GLAZING,glass_blue:GLAZING,glass_clear:GLAZING,water:{roughness:.05,metalness:0,probe:true,env:1.3},
 curtain:{roughness:.35,metalness:.25,probe:true,env:.7},
 steel:METAL,aluminum:METAL,silver:{roughness:.45,metalness:.45,probe:true,env:.6},zinc:{roughness:.42,metalness:.8,probe:true,env:.9},copper:{roughness:.4,metalness:.75,probe:true,env:.9},
 charcoal:{roughness:.7,metalness:.05},verdigris:{roughness:.7,metalness:.15},lamp:{roughness:.3,metalness:.5},
 white:PAINT,redpaint:PAINT,greenpaint:PAINT,yellow:PAINT,
 panel_dark:{roughness:.66,normal:1.2},panel_silver:{roughness:.3,probe:true,env:1,normal:1.2},
 asphalt:{roughness:.9,normal:1.3},
 concrete:{normal:1.3},stone:{normal:1.5},limestone:{normal:1.4},cast_stone:{normal:1.4},brick:{normal:1.5},brick_modular:{normal:1.6},brick_dark:{normal:1.6},
 stucco:{normal:1.3},stucco_red:{normal:1.3},render_white:RENDER,render_gray:RENDER,render_salmon:RENDER,
 timber_dark:{roughness:.7,normal:1.4},timber_tan:{roughness:.7,normal:1.4},
 terracotta:{roughness:.7},terracotta_panel:{roughness:.6,normal:1.3},terracotta_ribbed:{roughness:.6,normal:1.3},
 roofing:{roughness:.8,metalness:.1},
 // Venue surfaces: painted seat shells have a light sheen, turf and clay stay matte with relief, a video screen and
 // a glass-look bottle reflect like glazing.
 seats:{roughness:.55,normal:1.3},venue_concrete:{normal:1.2},turf:{roughness:.95},clay:{roughness:1,normal:1.2},padding:{roughness:.8},
 screen:GLAZING,bottle:{roughness:.15,metalness:0,probe:true,env:1},leather:{roughness:.7},
};
// Ground paving takes the albedo of worn, sunbaked concrete (about 0.3, warm grey): the facade concrete scan it shares
// averages 0.48, so pavements rendered near white and read as snow. Facades keep the scan's own albedo.
// The asphalt scan is yellow-brown (0.09/0.085/0.06); weathered road asphalt is a neutral grey near 0.12.
const GROUND=new Set(['terrain','sidewalks','cycleways','paths','steps','landscape','roads']);
const GROUND_FINISH:Record<string,Finish>={concrete:{tint:[.66,.63,.57]},stone:{tint:[.72,.69,.64]},venue_concrete:{tint:[.72,.7,.66]},asphalt:{tint:[1.3,1.35,1.55]}};
export function classOf(meshName:string){return meshName.split('___').pop()!.split('#')[0].split('__')[0];}
/** Applies the finish of each mesh's class to its materials (once per material) and returns the materials that
 *  should reflect a local probe of their surroundings, each carrying the reflection strength in userData. */
export function finishMaterials(meshes:THREE.Mesh[]){
 const done=new Set<THREE.Material>();const probe:THREE.MeshStandardMaterial[]=[];
 for(const mesh of meshes){
  const cls=classOf(mesh.name);const ground=GROUND.has(mesh.name.split('___')[0].split('__')[0]);
  const f=ground&&GROUND_FINISH[cls]?{...FINISH[cls],...GROUND_FINISH[cls]}:FINISH[cls];if(!f)continue;
  for(const m of Array.isArray(mesh.material)?mesh.material:[mesh.material]){
   if(done.has(m)||!(m instanceof THREE.MeshStandardMaterial))continue;done.add(m);
   if(f.roughness!==undefined)m.roughness=f.roughness;             // a factor on the roughness map where one exists
   if(f.metalness!==undefined&&!m.metalnessMap)m.metalness=f.metalness;
   if(f.normal&&m.normalMap)m.normalScale.setScalar(f.normal);
   if(f.tint)m.color.setRGB(f.tint[0],f.tint[1],f.tint[2]);
   if(f.probe){m.userData.envIntensity=f.env??.8;probe.push(m);}
   m.needsUpdate=true;
  }
 }
 return probe;
}
