/** Opaque room backing remains a large occluder even when hidden behind glass.
 * Dropping it from shadows turns a closed building into a window-grid lantern.
 */
export const NO_SUN_SHADOW = /^(glass|glass_dark|glass_light|glass_blue|glass_clear|curtain|screen|yellow)$/;
export const THIN_SHADOW_DETAIL = /^(aluminum|steel|silver|charcoal|white|chrome|seats)$/;

export function smallShadowDetail(surface: string, layer: string) {
  return ['facades', 'roofs', 'street_objects'].includes(layer)
    && (THIN_SHADOW_DETAIL.test(surface) || NO_SUN_SHADOW.test(surface));
}
