# Same-options mansard window A/B handoff

The candidate is staged outside `pipeline/`; no full release or frozen profile was changed. Baseline and candidate use the exact R31 snapshot profiles and roof observations from `dormer-pair-v2-raw-inputs/`, `--only [sf_201006_0017402, sf_201006_0018068]`, `--roof-shell`, and the same area. The generator emitted its normal raw GLB, façade lint, rule audit, and evidence for the two-feature section. The candidate adds only the staged mansard opening patch. The focused staged suite passed: 6 tests, including Earcut area checks for a polygon with a hole and a concave outline.

## Geometry delta

Compared source-space triangle signatures between the isolated baseline and candidate GLBs. For `sf_201006_0017402`, only `facades__sf_201006_0017402__observed_regen-sf_201006_0017402__s2p0___stucco#a9a59a` changes: 24 faces become 56 (12 exact faces retained, 12 removed, 44 added). Its source bounds are unchanged. The two measured-window cutouts cover 1.2349 m² of a 3.9000 m² projected mansard panel; 2.6652 m² (68.3%) of the panel remains, and the existing fascia triangles are retained. The separate 224-face `#e0d1a8` cornice is byte-for-byte geometrically unchanged.

For `sf_201006_0018068`, every source-space triangle signature is identical between baseline and candidate. Its real gable surface and edge treatment remain.

## Registered-camera result

For group 1 on `sf_201006_0017402`, the exact source-camera sample count is 162. With the actual cornice present, both baseline and candidate still have 0 glass-first rays: 144 hit the `#e0d1a8` cornice and 18 hit the `#f0ece0` frame. The candidate therefore exposes no additional pane area in the actual camera view by itself.

As a diagnostic only, omit the cornice node from the ray index: baseline then has 0 glass-first rays (124 hit the mansard, 38 the frame); candidate has 112 glass-first rays and 50 frame-first rays. This confirms the mansard cutouts are correctly placed and shows that cornice depth is the remaining visibility constraint. It is not a proposal to remove or flatten the cornice: the profile records it as a projecting 0.9 m cornice with dentils. The registered per-profile bound is 0.3105 m outward offset to clear all sampled pane rays, versus current maximum projection 0.9304 m (0.6199 m beyond that bound). The bound is camera/profile-specific and cannot justify a general projection reduction.

The exact next work is a depth/section review of this measured cornice and its dentils that preserves the source-measured projection while testing whether the underside/support profile can clear the source-visible dormer panes. Until then, keep this as an isolated tested section; do not repackage another street release.

## Artifacts

- `baseline-output/king-street-pilot.glb` and `candidate-output/king-street-pilot.glb`: same-options A/B raw sections.
- `baseline/` and `candidate/`: isolated copies of the R31 pipeline; the proposed code exists only in the candidate copy.
- `test_facade_roof_detail_height_frame.py`: staged focused suite.
- `candidate-group-ray-audit.json`: full-window registered source-camera group audit for the staged candidate.
- `../dormer-pair-v2-raw-inputs/`: exact frozen input snapshots used for the section.
