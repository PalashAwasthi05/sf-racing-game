from pathlib import Path
p=Path(__file__).resolve().parents[1]/'outputs/world-generation-blueprint.md'
s=p.read_text()
addition='''

## Imagery discovery and surface replacement contracts (revision 0.12)

Discover imagery across the complete requested area before selecting individual views. For dense providers, use spatial coverage indexes and filter metadata locally instead of repeatedly requesting an unbounded history for a large rectangle. Cache image hashes and metadata. Within small spatial cells, rank capture date first and favor panoramas among similarly dated images; under a capture budget, distribute selections spatially. Then refine selection using facade visibility, occlusion and resolution. A coverage index is not proof of facade visibility. Historical-only cells remain historical-only.

Rank available sources by capture date, usable pixel resolution, viewpoint, calibration and permitted reuse. A provider brand does not guarantee the newest observation for a particular surface. Google Maps can be consulted for comparison; reusable reconstruction inputs need appropriate permission. Do not treat a screenshot as permission to extract a reconstruction dataset. Mapillary acquisition uses a runtime access credential, never a source-file secret.

Normalize image hashes, dates, camera model and source links into the input evidence index. Check supplied poses against their reconstruction records using capture time, rotation and camera-center agreement. Keep provider consistency separate from alignment to the metric world. Require independently validated world registration before admitting image-derived wall measurements. Pose metadata alone must not pass this gate. Multiple frames from one capture sequence do not automatically count as independent evidence.

Every surface replacement is a transaction over an explicit local surface domain. Compute the expected host minus owned apertures, compare that against the emitted front-facing triangles on the declared plane, and reject missing area, overlapping faces or geometry outside the domain. An intentional opening must have an explicit owner. Preserve the host outside the exact emitted-and-owned-opening cut domain. Do not remove a whole frontage or a broad roof strip merely because some replacement exists nearby. A planar front-skin audit does not certify rear faces, returns, soffits or the full solid; those require their own domains and support checks.

For inherited geometry, a conservative roof repair can recover measured omissions: intersect the owner footprint minus existing upward surfaces with point-supported cells that agree with an adjacent roof plane. Keep footprint holes. Bound recovery distance, reject unsupported or disconnected cells, and emit the new top on precisely the same plane. Preserve the neighboring material and metric UV scale. This restores supported missing coverage; it does not prove source freshness, recover arbitrary facade holes or invent surfaces across unknown voids.

Add adversarial regression cases for removed and reversed triangles, duplicate coverage, intentional apertures, narrow strips, shifted frames, courtyards, different-height observations, sloped repairs, unsupported gaps and source-pose mismatch. After changes, regenerate the affected section and compare the same camera views, then run the rule over the full area. Unresolved regions remain recorded; a passing synthetic test is not a complete visual acceptance result.
'''
if 'revision 0.12' not in s:p.write_text(s+addition)
