import type {Box3, Matrix4} from 'three';

// Relative to the viewport so a change of display resolution doesn't change
// which buildings appear. Nearby owners are also admitted by metric distance.
export const detailScreenFraction = 1 / 12;
const corners = new Float64Array(24);

/** Fraction covered by the projected bounds, clipped to the actual viewport.
 * Project near-plane intersections as well: projecting a corner behind the
 * camera directly can turn a sliver at the screen edge into a full-screen box. */
export function frontageScreenFraction(box: Box3, clip: Matrix4, near: number) {
  const m = clip.elements;
  let left = 1, right = -1, bottom = 1, top = -1, inFront = 0;
  for (let i = 0; i < 8; i++) {
    const x = i & 1 ? box.max.x : box.min.x;
    const y = i & 2 ? box.max.y : box.min.y;
    const z = i & 4 ? box.max.z : box.min.z;
    const cx = m[0]*x + m[4]*y + m[8]*z + m[12];
    const cy = m[1]*x + m[5]*y + m[9]*z + m[13];
    const w = m[3]*x + m[7]*y + m[11]*z + m[15];
    corners[i*3] = cx; corners[i*3+1] = cy; corners[i*3+2] = w;
    if (w < near) continue;
    inFront++; left = Math.min(left, cx/w); right = Math.max(right, cx/w);
    bottom = Math.min(bottom, cy/w); top = Math.max(top, cy/w);
  }
  if (!inFront) return 0;
  if (inFront < 8) for (let i = 0; i < 8; i++) for (let axis = 1; axis <= 4; axis *= 2) {
    if (i & axis) continue;
    const a = i*3, b = (i|axis)*3, wa = corners[a+2], wb = corners[b+2];
    if ((wa < near) === (wb < near)) continue;
    const t = (near-wa)/(wb-wa);
    const x = (corners[a] + t*(corners[b]-corners[a]))/near;
    const y = (corners[a+1] + t*(corners[b+1]-corners[a+1]))/near;
    left = Math.min(left,x); right = Math.max(right,x); bottom = Math.min(bottom,y); top = Math.max(top,y);
  }
  return .25 * Math.max(0, Math.min(1,right)-Math.max(-1,left)) *
    Math.max(0, Math.min(1,top)-Math.max(-1,bottom));
}
