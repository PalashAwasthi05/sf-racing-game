import * as THREE from 'three';

const SOLAR_BEGIN =
  '#if ( NUM_DIR_LIGHTS > 0 ) && defined( RE_Direct ) && defined( USE_CSM ) && defined( CSM_CASCADES )';
const SOLAR_END =
  '#endif\n\n\n#if ( NUM_DIR_LIGHTS > 0 ) && defined( RE_Direct ) && !defined( USE_CSM ) && !defined( CSM_CASCADES )';
const SOLAR_SNAPSHOT = /* glsl */ `
#ifdef USE_CLEARCOAT
 vec3 coatedBeforeSunBase=reflectedLight.directSpecular;
 vec3 coatedBeforeSunCoat=clearcoatSpecularDirect;
#endif
`;
// A photographic shoulder belongs to the broad lobe, before the two layers are
// composed. A radiance derivative cannot identify a glint: a broad reflection
// can have a steep edge on a curved panel. Actual roughness identifies its width.
// Only the directional solar increment is compressed; point lights and the
// separate street-lamp/headlight injection keep their original radiance.
const SOLAR_SHOULDER = /* glsl */ `
#ifdef USE_CLEARCOAT
 vec3 broadSun=max(reflectedLight.directSpecular-coatedBeforeSunBase,vec3(0.));
 float broadPeak=max(broadSun.r,max(broadSun.g,broadSun.b));
 float broadShare=smoothstep(.07,.18,material.roughness);
 reflectedLight.directSpecular=coatedBeforeSunBase+broadSun*mix(1.,1./(1.+broadPeak/.055),broadShare);
 vec3 coatSun=max(clearcoatSpecularDirect-coatedBeforeSunCoat,vec3(0.));
 float coatPeak=max(coatSun.r,max(coatSun.g,coatSun.b));
 float broadCoatShare=smoothstep(.07,.18,material.clearcoatRoughness);
 clearcoatSpecularDirect=coatedBeforeSunCoat+coatSun*mix(1.,1./(1.+coatPeak/.12),broadCoatShare);
#endif
`;

/** Run after the CSM include is expanded, before the material's final lighting composition. */
export function patchCoatedLight(
  shader: THREE.WebGLProgramParametersWithUniforms,
) {
  const before = 'ReflectedLight prevLight = reflectedLight;';
  const blend =
    'reflectedLight.directSpecular = mix( prevLight.directSpecular, reflectedLight.directSpecular, blendRatio );';
  const physical = THREE.ShaderChunk.lights_physical_pars_fragment;
  const coatStart = physical.indexOf('vec3 BRDF_GGX_Clearcoat('),
    coatEnd = physical.indexOf('vec3 BRDF_GGX(', coatStart);
  const coat = physical.slice(coatStart, coatEnd);
  const distribution = 'float D = D_GGX( alpha, dotNH );';
  if (
    !shader.fragmentShader.includes(before) ||
    !shader.fragmentShader.includes(blend) ||
    !shader.fragmentShader.includes(SOLAR_BEGIN) ||
    !shader.fragmentShader.includes(SOLAR_END) ||
    !shader.fragmentShader.includes(
      '#include <lights_physical_pars_fragment>',
    ) ||
    coatStart < 0 ||
    coatEnd < coatStart ||
    !coat.includes(distribution)
  )
    throw new Error(
      'Coated lighting expected the r180 CSM fade and physical lighting includes',
    );
  // A normalized Beckmann distribution models the polished film's Gaussian
  // microfacet slopes. GGX's long tails produced a broad white veil around the
  // narrow direct coat reflection; the Gaussian keeps the bright compact core.
  // The coloured base layer and the sampled environment retain their GGX BRDF.
  const polishedCoat = coat.replace(
    distribution,
    /* glsl */ `
 float coatAlpha2=max(alpha*alpha,1.e-8);
 float coatCos2=max(dotNH*dotNH,1.e-5);
 float D=RECIPROCAL_PI*exp((coatCos2-1.)/(coatAlpha2*coatCos2))/(coatAlpha2*coatCos2*coatCos2);`,
  );
  // r180 fades ReflectedLight, but clearcoat accumulates in a separate global.
  // Blend that accumulator too, so overlap receives one solar coat reflection.
  shader.fragmentShader = shader.fragmentShader
    .replace(
      before,
      before +
        '\n#ifdef USE_CLEARCOAT\nvec3 previousCoatDirect = clearcoatSpecularDirect;\n#endif',
    )
    .replace(
      blend,
      blend +
        '\n#ifdef USE_CLEARCOAT\nclearcoatSpecularDirect = mix(previousCoatDirect, clearcoatSpecularDirect, blendRatio);\n#endif',
    )
    .replace(SOLAR_BEGIN, SOLAR_BEGIN + '\n' + SOLAR_SNAPSHOT)
    .replace(SOLAR_END, SOLAR_SHOULDER + '\n' + SOLAR_END)
    .replace(
      '#include <lights_physical_pars_fragment>',
      physical.replace(coat, polishedCoat),
    );
}
