from pathlib import Path
p=Path('outputs/world-generation-blueprint.md');s=p.read_text().replace('Version 0.4 · September 8, 2026 · Street-first inverse facade modeling','Version 0.5 · September 8, 2026 · Building form before window repetition')
a=s.index('**Revision 0.4:');b=s.index('**Budget rule for the next scale test**')
new='''**Revision 0.5: infer a connected building envelope before its details**

The previous representation was too weak: it could vary window grids and colors inside a flat wall, but still flattened projecting blocks, lower wings, separate roof cores and open structure. More windows could not fix that. A building must be represented by a hierarchy of surfaces and attached assemblies, with changing height and depth. “Floors” are not a universal division: two double-height loft bands may contain many transoms and mezzanines.

The new unit of inference is the **connected street-visible building envelope**. Its parts have their own horizontal spans, height intervals, surface orientations, signed offsets, material fields, opening shapes and attached components. Neighboring frontages can share explicit corner and roof constraints. A child projecting surface replaces its parent's skin over that area; it is not pasted over an opaque full-height block. Sky gaps and open steelwork remain empty space.

1. **Choose observations by visible structural uncertainty.** Sample approaching roads, intersections and side glances at driver height. First ask which images resolve the outer contour, material boundaries, setbacks, projecting blocks and voids. Favor a near-frontal image plus an oblique view that exposes returns. A detailed window crop cannot resolve a hidden roof setback. Batch imagery across adjacent buildings, deduplicate by viewpoint and date, and keep actual capture dates separate from file publication dates.
2. **Recover the building's organization.** Annotate contiguous masses and their relationships: supports, adjacent wings, higher/lower cores, inset connectors, projecting glass, free-standing shades, roof solids, voids and exposed frames. Mark the locations where material, normal, height or opening topology changes. Window transoms are initially only glazing divisions, not inferred floor slabs. Preserve exceptional polygonal openings and nonuniform vertical intervals. Use roof and facade LiDAR as depth/height evidence, and street imagery as form and appearance evidence.
3. **Fit structural constraints before repeat patterns.** Give each annotated roof family a frontage interval and a bounded depth/height search region. Fit a robust local height mode to the relevant LiDAR samples. A tall background tower or stadium light support must not set the low street wing's height. Share only named datums supported as continuous across views; do not merge all similar-looking horizontal lines. Solve identified corners from the adjacent facade planes, accepting only bounded adjustments and recording the displacement. Inconsistent observations remain unresolved.
4. **Describe architecture with a compact graph.** A node contains its parent, extent, material and signed depth. Children use coordinates relative to the parent, so the same structure survives changes of scale and orientation. Repeat nodes hold counts or sparse fitted centers rather than one record per window. Attached components include section-swept cornices, detached blades on posts, frames, fins, pyramidal roofs, circular clock assemblies and braced frames. Those are reusable operators; addresses and landmark names are restricted to input matching. A stadium is assembled from these operators, not selected by an address-specific mesh function.
5. **Fit openings within each surface.** Retain projective rectification and robust repeat fitting for locally planar regions. Assign separate glazing grammars to commercial bases, punched-window wings, ribbon windows, loft bands and glass corners. Fine subdivisions and operable panes live inside a larger opening. Repetition may complete occluded cells only within the same supported region. Material joints stop at openings and region boundaries. Shape evidence chooses the topology; uncertain metric depths retain their estimate status.
6. **Compile replacement geometry.** Remove the previous wall and roof shell only within each reconstructed frontage's replacement volume. Cut child regions and apertures from parent surfaces. Build returns, soffits and recessed glass on distinct planes; sweep the observed cornice section; build roof blades and steel braces as separate members. Keep the background LiDAR mass where it has not been reconstructed. Do not leave an opaque block inside an observed open gap. Surface textures supply grain, not the outline of major architecture.
7. **Test structural likeness before decorative density.** Compare matched driver views under fixed lighting. Review the projected silhouette, height changes, sign of depth offsets, material-zone boundaries, aperture outlines, corner continuity and visible voids before adding wear, signs or clutter. Then test different oblique views to expose incorrect depth. Keep held-out image measurements separate from fitting images. A facade fails this gate if its characteristic massing or opening topology is wrong, even if its window count is high.
8. **Cache and measure the expensive part.** Store observations and graph revisions once, keyed by source/date and geometry version. Numerical fitting and mesh generation use no language-model calls. In the scaled design, a vision pass returns a short graph delta for changed regions; component recipes and repeated cells expand locally. Current implementation still rebuilds the whole pilot, and bulk automatic image parsing and selective tile rebuilds remain future work. Record acquisition, annotation and correction time separately from cached generation time.

The implementation deliberately distinguishes **manual image interpretation**, **bounded numerical fitting**, and **synthetic completion**. This pass does not pretend that a fully automatic multi-view facade recognizer exists. The hierarchy was annotated by the assistant from photographs; local roof levels and explicit cross-facade constraints are computed. The repeat fitter from v0.4 remains in place for the earlier loft and hotel banks.

**Applied examples and evidence**

- **The Beacon, northwest of Third/King:** a higher corner with ribbon glazing and a glass prow, a separate raised pale spine, a narrow recessed glass field, lower punched-window wings, contrasting commercial base and projecting cornices. The exact corner is visible in the [JLL property brochure](https://property.jll.com/listings/the-beacon-retail-200-298-king-st-south-of-market) and the January 2025 panorama. A previously collected photograph of a different Beacon corner was not transferred here.
- **Glassworks, southwest of Third/King:** three lower commercial glazing bands, two double-height upper loft bands, projecting silver frames, separate corner glazing, terracotta piers/core and a detached roof blade. The [architect's actual photographs](https://www.perryarchitectsinc.com/glassworks) establish those relationships. Historic detail images were compared with January 2025 Street View for continuing macro-form; this is not verification of every current pane or storefront. The building is not China Basin Landing farther south.
- **Oracle Park:** masonry wings and towers, projecting trim and piers, clock faces, pyramidal roofs, open braced stand structure and light banks. The [stadium technology partner photograph](https://onediversified.com/hubfs/Diversified_San%20Francisco%20Giants%20_Case%20Study.pdf?hsLang=en) resolves tower relief. The [operator documents 122-foot towers and 45-foot flagpoles](https://www.mlb.com/giants/ballpark/information/guide). Dimensions of other members remain estimates; current signs and sculpture are not reconstructed.

The [January 2025 Street View panorama at Third and King](https://www.google.com/maps/@37.7781797,-122.3919934,3a,100y,265h,110t/data=!3m4!1e1!3m2!1szpKsGNDss7FrpMAioZXjSw!2e0) was inspected at headings 265°, 190° and 85° for the three building identities and major form. Source pixels are not embedded as textures. This panorama and the historic detail photographs are evidence of different ages, not an all-current photographic survey.

**What revision 0.5 actually implements**

| Stage | Pilot status |
|---|---|
| Image parsing | Assistant-authored hierarchical observations for three additional building identities; not a bulk automatic detector. |
| Structural model | Eight oriented envelope profiles; nested material/depth regions, multiple roof levels, arbitrary polygonal apertures and independent attached assemblies. |
| Numerical fitting | Bounded local LiDAR roof-family fitting, explicitly shared height datums and bounded corner joins. Depth values remain visual estimates. |
| Earlier inference | Five earlier frontage grammars and two sparse numerical opening-bank fits remain available. |
| Generation | Generic envelope compiler replaces old frontage shells, generates apertures/returns, metric cladding joints, roof profiles, awning panes and open structures. |
| Rendering | Existing AgX daylight, sky environment, camera-following sun shadows and GTAO retained. Street viewpoints now include the new targets. |
| Validation | Background-height rejection, rotation/translation invariance, parent/child coordinates and depth, shared corner/datums, nonrectangular apertures and geometry checks. Independent matched-photo accuracy and runtime performance remain unmeasured. |

The wider scene still contains many unverified facades. Depth uncertainty, occluded areas, sculptures, exact commercial signs, building-specific material scans, calibrated camera poses and a fully reconstructed stadium bowl remain gaps. Image hierarchy is much richer than a flat repeated wall, but it does not by itself establish the user's requested realism. The 1:1 accuracy gate remains open.

**Structural acceptance gate for subsequent revisions**

For each visible building, compare at least two independently positioned views and record whether the main silhouette breaks, projecting/inset volumes, material regions, roof solids/voids and irregular aperture families are supported. Use silhouette overlap and boundary distance, feature reprojection error, signed depth consistency and world-space corner gaps; set numerical thresholds from a calibrated validation block before treating them as citywide pass criteria. Give a structural mismatch priority over adding more repeated decorative geometry. Keep glass reflections and illumination out of the geometry error metric as far as possible.

No numeric perception claim such as “90% of realism” is assumed. The user's priority is implemented by evaluating street-visible building structure ahead of decorative density. The next portability test must include unfamiliar stepped buildings, curved or nonorthogonal surfaces, deep balconies and ornament; a renderer that can draw a component is not proof that the observation pipeline can infer it reliably.

'''
s=s[:a]+new+s[b:];p.write_text(s)
p=Path('work/pipeline/README.md');s=p.read_text();s='''# Street-first reconstruction pipeline v0.5

`envelope-observations.json` stores building matches and image-derived hierarchical structure. `envelopes.py` fits bounded roof-family levels, joins explicitly corresponding corners/datums, resolves nested regions, and compiles replacement facade geometry. Geographic identifiers occur only in observation data. Source photos are research references and are not shipped as textures.

The graph is assistant-annotated, not an automatic vision output. Signed depths, fine dimensions and many glazing details remain estimates. `validate_envelopes.py` checks geometry and structural invariants on a synthetic unfamiliar building. It does not certify photographic likeness. Reuse requires new observation data for new buildings; do not add building-specific compiler branches.

Run from the bundle root with the geospatial Python requirements installed:

```sh
python work/validate_envelopes.py
python work/build_world.py
python work/validate_detail.py
```

The complete generator also needs the earlier geodata and LiDAR fetch steps below. Those cached source datasets are not included in the bundle.

---

'''+s;p.write_text(s)
