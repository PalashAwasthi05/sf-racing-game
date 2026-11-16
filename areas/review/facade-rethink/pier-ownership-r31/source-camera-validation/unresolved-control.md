# Unchanged control: `sf_201006_0017402__s2p0`, group 1

The 0/162 expected-receiver result is a real source/geometry occlusion, not a glazing-classifier miss. The registered crop shows two unobstructed dormer windows; their exact group opening center rays project to photo pixels about `(333,72)` and `(507,90)`. See `group-1-center-rays.jpg`.

## What blocks the rays

On both R30 and raw R31, the center rays hit the profile's `#e0d1a8` cornice, then the generated mansard face, then glass. Across the 162 group-1 interior samples, first-hit counts are 144 cornice and 18 `#f0ece0` frame; neither category is a glazing hit:

- Cornice hit: triangle 90 of `facades__sf_201006_0017402__observed_regen-sf_201006_0017402__s2p0___#e0d1a8` (global triangle 4442 in the R31 raw feature ray index). Its vertices in profile coordinates `(u, V, z)` are `(-359.13348,-203.40202,-22.13949)`, `(-353.54361,-203.40202,-21.69949)`, and `(-359.13348,-203.40202,-21.69949)`. The source facade plane is `V=-204.315`, so this top face projects about **0.913 m** outward. The center rays hit it at `z=-21.85590`, and then hit its top at `z=-21.69949`.
- Window opening range is `z=-21.64254..-20.95646`. The cornice is therefore wholly below the opening; it occludes the view **from below** because of its projection. It does not geometrically cross the window's height range.
- After the cornice, each center ray hits two `stucco#a9a59a` triangles (global 3973/3975 for the first pane) from the profile's generic mansard roof. The first roof-face hit is at `z=-21.28249`, inside the window opening, at `V=-204.332` (about 0.017 m behind the measured wall plane). The source reading places this mansard over `x=0..3.5 m, z=7.4..8.4 m`; that extent covers both measured dormer cells. This roof face intrinsically intersects the window area because the reading records the two dormers as unmodelled.
- The ray then reaches `glass` triangles at `z≈-21.21`, `V≈-204.45`, with interior behind them. **The glazing exists behind both occluders.** The intended `glass` material has roughness 0.06; the cornice and mansard are distinct opaque geometry. No tree or unrelated site object is responsible.

The host-shell audit reports no base-shell-first rays. R31 raw preserves the same cornice and mansard blockers as R30, so this unchanged profile is not a pier-cut regression.

## Projection bound supported by this camera

For this one profile, the source hash and accepted `skyline` registration are valid, the registered side matches the profile's image and metric frame, and the projected opening centers land on the two visible panes. Sampling 162 interior points across both registered group openings and tracing them to the top of the cornice gives a minimum outward ray offset of **0.3105 m**. This is a profile-specific conservative upper bound on cornice projection at its top face if every sampled visible pane point must remain unobstructed. The current generated section reaches about 0.93 m. The bound follows from the registered pose plus this profile's mapped opening geometry; it is not a transferable sitewide constant.

Reducing the cornice below 0.3105 m would clear those sampled camera rays from the cornice, but **would not clear the mansard face**, which intersects the panes at the wall. A correct repair needs local dormer/roof ownership or an explicit abstention when a generic mansard crosses measured dormer openings; a global projection clamp would be insufficient and is not recommended from this single profile.
