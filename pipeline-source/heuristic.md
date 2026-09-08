# Evidence-based world generation: San Francisco pilot

Version 0.4 · September 8, 2026 · Street-first inverse facade modeling

Build a reusable pipeline that reconstructs a racing-game environment from observations, with language models used primarily to develop the pipeline and diagnose exceptions. The user's benchmark is one person, one existing AI subscription, and one calendar month for all of San Francisco. Paid data is acceptable; the initial use is a private prototype.

**Feasibility judgment:** citywide generation at real scale is a reasonable engineering experiment. Complete, current, accurate reproduction of every facade, marking, sign, tree, and minor detail is not established as achievable within this budget. Missing or stale observations remain missing or stale after generation. A convincing approximation must not be reported as completion of the exact specification.

The main unknown is the availability of recent, sufficiently detailed, licensed street-level observations across the entire requested area. Another is the labor required to turn those observations into clean, drivable game assets. The first-week pilot must measure both.

**Revision 0.4: optimize the view from the road**

A facade with many windows can still be the wrong facade. The previous pass produced excessive repetition, applied incomplete observations too broadly, and compressed some facade layouts into low courtyard roof heights. The revised algorithm fits each wall's visible organization before creating fine geometry. It treats facade inference as an inverse problem: choose a compact description that explains the observations, rather than choose a plausible style and populate a footprint.

The unit of work is a street-visible wall region. A single wall may have an entrance zone, tall loft windows, projecting terrace frames, a blank service section, and a separate glazed crown. Each region gets its own layout and depth hypothesis. Do not force a single floor pitch or column count across these transitions.

1. **Define the driving view distribution.** Sample the road network in both travel directions at 5–10 m intervals, with a 1.25 m default driver eye height and a documented field of view. Include approaching intersections and oblique side views. Cover all drivable streets when estimating the citywide target; focusing the pilot on one street is a measurement choice, not a replacement for city coverage. Rank missing work by projected visible error, proximity, time in view, and uncertainty. Courtyards visible only from overhead do not compete equally with entrances two meters from the camera.
2. **Reconcile identities and facade planes.** Match current building identities to older municipal footprint parts. When several old parts refer to the same contemporary building and overlap sufficiently, consolidate them into one documented identity envelope. Record disagreement instead of silently keeping overlapping buildings. Group collinear wall edges and give them one coordinate system: the facade pattern must not restart at every footprint vertex. Distinguish the main street wall, roof machinery, low courtyard slabs and upper setbacks; global minimum roof height cannot set the height of a street facade.
3. **Acquire evidence in frontage batches.** Prefer images that see several adjacent walls, then deduplicate by pose, time, resolution and newly visible uncertain area. Google Street View is useful for dated targeted observations; operator, architect and engineer photographs can clarify construction details. Reject design renderings as as-built evidence. Reject a courtyard view as proof of street-facing bay counts. Source upload dates are not capture dates. Use a primary source's textual dimension as dated supporting evidence, not a fresh survey.
4. **Annotate once, fit cheaply.** A vision pass supplies the wall-region quadrilateral, a sparse selection of opening centers or corners, occlusion masks, semantic region boundaries, material families and topology. Store pixel coordinates and source identity as compact data. Compute the planar homography numerically. Search for the simplest row/column lattice that explains the rectified observations, with a robust residual term, constraints from opening sizes and spacing, and a complexity cost that penalizes spurious extra rows. Keep exceptional windows and blank fields outside the repeat rule. Cache the packet and fit by content hash; regenerating geometry does not reread images or use language-model calls.
5. **Complete only within supported structure.** Propagate a fitted repeat into masked cells within the same observed wall region. Mark those cells as inferred repeats. Moving the pattern to a different facade is a separate transfer hypothesis, even on the same building. Preserve uncertainty in absolute scale, depth, wall association and capture age. Low fit error in the input photograph is not proof of 3D accuracy. Escalate poor fits, non-planar bays and ornament to another observation or an exception job.
6. **Compile the layout into architecture.** Partition walls around the openings. Build reveals, glazing at the back of the recess, structural frames, terrace floors/rails, mullions, transoms, projecting bay cheeks, cladding panels, shade fins, ground-floor doors, roll-up entries and canopies. Use independently specified topology for each window family. Keep an opaque dark interior plane behind glazing for this prototype; it is a depth cue, not a recovered room. Scanned PBR surfaces supply material grain, while physical UV scale and correct linear color factors preserve their response to light.
7. **Keep street surfaces continuous.** Densify road and sidewalk surfaces against the same terrain triangulation used by the ground and camera. Interpolating a long road polygon only at its corners can let terrain rise through it. Connect curb transitions and markings to the same vertical reference. Street inventories seed attachment candidates; they do not establish exact curb lines or current sign positions.
8. **Judge form under controlled lighting.** Use a fixed driver camera and fixed lighting for correspondence comparisons. The viewer offers afternoon sun and soft daylight, a shared sky environment, smaller camera-following shadow coverage and local ambient occlusion. Correct sRGB-to-linear material factors before compensating with exposure. Avoid excessive ambient fill that erases recesses. See the [glTF material specification](https://registry.khronos.org/glTF/specs/2.0/glTF-2.0.html#materials) and [Three.js GTAO](https://threejs.org/docs/pages/GTAOPass.html) for the material and shading conventions. These are raster lighting approximations, not calibrated radiometry, baked global illumination or path tracing. A more attractive sun angle cannot compensate for a wrong opening layout.
9. **Validate held-out evidence and cost separately.** Withhold some landmarks to catch fitting overreach; then acquire genuinely independent views to test geometry. Measure visible outline and opening reprojection error, curb and sign displacement, image age, material mismatch and supported coverage. Keep image-pixel residuals distinct from metric errors. Log observation labor, fit time, regeneration time, transfer size, triangles, memory and frame time. The one-person-month claim depends on the observation and correction workload as much as model generation speed.

The repeat fitting approach follows the general research idea that facade symmetry and architectural constraints help interpretation under occlusion; the implementation here is a smaller sparse-landmark optimizer, not a reproduction or performance claim for those research systems. [Symmetry-aware facade parsing](https://www.microsoft.com/en-us/research/publication/symmetry-aware-facade-parsing-with-occlusions/), [inverse procedural facade layouts](https://arxiv.org/abs/1308.0419)

**What revision 0.4 implements**

| Stage | Actual pilot status |
|---|---|
| Image observation | Assistant-authored sparse pixel landmarks for two banks; visual grammar records for additional facade regions. No automatic bulk detector. |
| Numerical inference | Projective rectification and robust search for row/column spacing; completed repeat cells retain their status. |
| Applied evidence | Five oriented frontage profiles across four building identities; two numerically fitted banks, with the other layouts marked as visual grammars or same-building transfer. |
| Current dates | Two buildings use January 2025 Street View observations. Two hotel identities use photographs with unknown capture dates. No claim of 2026 facade freshness. |
| Facade geometry | Shared region/opening compiler, distinct ground floors, terrace frames, projected bays, shade fins, glazing topology and facade material zones. |
| Terrain and identities | Reconciled fragmented identities; surface densification and common ground interpolation; frontage height taken from local high-level LiDAR support rather than a low courtyard slab. |
| Lighting | AgX tone mapping, reduced ambient fill, camera-following sun shadows, GTAO contact shading, sun/soft daylight comparison, and corrected material color space. |
| Validation | Synthetic perspective/occlusion recovery; non-overlapping opening regions; same-image held-out landmarks; geometry and transfer checks. Independent matched-photo street accuracy and browser performance remain unmeasured. |

The loft fit recovers a 5 × 3 frame bank; the hotel bay-bank fit recovers 3 × 7. Two same-image held-out centers per bank give errors of approximately 4.6/9.4 pixels and 11.2/16.1 pixels respectively. These are approximate landmark consistency errors, with analyst-selected region corners. They are not independent image validation or centimeter-level measurements. Three separate synthetic perspective/noise/occlusion cases recover their expected 5 × 4, 3 × 7 and 8 × 2 grids, with held-out errors below one pixel; those synthetic checks establish only that the fitter works on its assumed model.

Input files contain the location-specific evidence; the fitter and component compiler contain no address-specific geometry branches. A Mapillary metadata adapter is still available but has not been run with account credentials. Dense automated image extraction, multi-view metric calibration, sculpted ornament recovery and material capture of the actual walls remain unimplemented. Most of the wider pilot still uses unverified priors. The 1:1 accuracy gate remains open.

**Budget rule for the next scale test**

Measure the next 20 frontages using the same schema, including unfamiliar facade types, without changing the geometry compiler. Record separately: image acquisition/annotation minutes; exception correction minutes; cached numerical fitting seconds; asset generation seconds; and image/model token usage. If a frontage still requires custom mesh coding, count it as an algorithm failure rather than hiding it in a one-off fix. Do not extrapolate two fitted banks to an entire city.

The desired citywide loop is one image packet per visible frontage family, one numerical fit per changed packet, shared component assets, and sparse exception review. An existing AI subscription is not evidence of unlimited image throughput or included paid API processing. Bulk vision costs and usable capture coverage still need measurement. The current prototype does not demonstrate completion of San Francisco at the requested fidelity within one month.

The model still bakes repeated geometry into ordinary glTF meshes. Gzip transfer chunks reduce download size, but do not provide spatial streaming or HLOD. Instancing, collision and measured runtime optimization remain necessary before city-scale generation. Proposed acceptance targets remain 95% usable dated street-visible coverage, independent depth-bearing observations, 0.15 m opening/curb positions and 0.30 m sign positions, tuned and then held fixed against actual driving-camera comparisons. None of these targets is reported as passed.

**Preventing location-specific fixes**

Keep source adapters, observation records, fitting, component generation and rendering separate. There must be no address ID, landmark name or target coordinate in a geometry-selection branch. Geographic identifiers belong in input matching. Regional road standards and architecture priors are configuration with provenance. A correction should either add a reusable component, improve a shared fit, fix source reconciliation, or supply new evidence through the existing schema.

Test portability by changing the region and using a withheld neighborhood with no new compiler branches. A system that needs manual facade coding for each new block fails the one-month target even if this pilot looks good. The appropriate citywide estimate is observed correction minutes per wall/street segment plus automated processing and data-acquisition time—not tokens spent to produce one polished image.

**What “1:1” means**

Separate geographic scale, geometric accuracy, visual correspondence, completeness, and freshness. A world can have one real meter per world meter while containing incorrect buildings. Record and evaluate these dimensions separately.

Use the city boundary as the initial coverage contract, including its islands. Explicitly inventory disconnected areas and bridge crossings; the airport and the opposite ends of major bridges require boundary decisions because a recognizable metropolitan scene can extend beyond the city. Do not quietly substitute downtown or a few racing routes for the full city. The Census lists 46.91 square miles of land, approximately 121.5 km². Water and boundary tiles add processing extent. [Census land area](https://www.census.gov/quickfacts/fact/csv/sanfranciscocitycalifornia/LND110220)

Set an explicit snapshot date. For a contemporary build, store the last actual observation date of each attribute and flag stale or unknown attributes. A 2026 file publication does not make a 2023 scan current. A permit indicates potential change, not a completed building. Cars, pedestrians, temporary works, water motion, illumination, and weather require either a specified captured moment or an explicitly simulated state.

**Data sources and their roles**

| Layer | Initial source | What it contributes | What remains unresolved |
|---|---|---|---|
| Ground and above-ground geometry | 2023 USGS San Francisco LiDAR | Terrain, roof evidence, canopy, and some infrastructure geometry | Later changes, occlusions, facade detail, thin features |
| Terrain baseline | USGS 0.25 m DEM from that survey | Consistent elevation starting point | Grid spacing is not positional accuracy; bridge decks need separate handling |
| Building identity and candidate outlines | DataSF footprints, Overture, OSM | Matching records and reconstruction constraints | Roofprints versus wall footprints, old outlines, missing heights |
| Road connectivity | City street centerlines and OSM | Segment identity, connections, street names, available traffic attributes | Lane boundaries, curb geometry, exact road surface |
| Street envelope | City right-of-way polygons | Broad spatial constraint | Public right of way is not the drivable pavement footprint |
| Lanes and markings | Recent licensed true orthophotos, street imagery, SFMTA striping documents | Observed paint, curb use, lane configuration, design dimensions | Occlusion, faded paint, drawing registration, design versus installed state |
| Signs and signals | City inventories, corroborated by imagery | Candidate type and location | Exact pole position, orientation, sign panel text, completeness |
| Trees | DataSF Street Tree List and LiDAR canopy | Tree candidates and vegetation extent | Inventory date, exact shape, private and park trees |
| Facades and street-detail evidence | Mapillary sequences plus licensed street panoramas and obliques | Surface appearance, visible architectural details, and corroborating views | Coverage, freshness, camera pose, hidden surfaces, derivation rights |
| Targeted visual reference candidate | Google Maps / Street View, subject to the applicable use permissions | Potentially newer views of facade changes, street layouts, and missing objects | Reference viewing, screenshot use, automated extraction, and texture reuse require separate assessment; no default bulk ingestion |
| Rapid visual foundation | Licensed textured reality mesh | Existing textured geometry where available | Capture age, ground-view quality, holes, baked lighting and objects, coverage |
| Updates | New captures, municipal change records, permits | Candidate changed regions | Independent confirmation of what physically changed |

The 2023 survey's point-cloud record offers spatially queryable Entwine Point Tiles. Query the region needed for each processing job rather than loading the whole survey into memory. The point-cloud metadata describes the access path and its coordinate treatment. [USGS LiDAR access through NOAA](https://www.fisheries.noaa.gov/inport/item/73386/full-list)

The related DEM is 0.25 m, and its source survey was captured April 20, 2023. Original deliverables used NAD83(2011) San Francisco CS13 meters and NAVD88/Geoid18 elevations. Read the metadata for the actual distribution downloaded: a republished point-cloud product can use a different horizontal or vertical representation. [DEM metadata](https://www.fisheries.noaa.gov/inport/item/73506)

DataSF's footprint description traces its geometry to a 2010 Pictometry model with later refinement. Treat it as an identity and outline candidate, not proof of current silhouettes. [Footprint catalog](https://dev.socrata.com/foundry/data.sfgov.org/ynuv-fyni/embed)

Overture supplies building and building-part records with optional heights and source information. It is useful for reconciliation and geographic expansion, but an updated release is not a new survey of every building. Its buildings theme carries ODbL. [Overture buildings guide](https://docs.overturemaps.org/guides/buildings/)

The city's sign inventory explicitly says many locations are assigned to an intersection or midblock point, and the inventory may be incomplete. These points must not become exact pole placements without corroboration. [Street Signs metadata](https://catalog.data.gov/dataset/street-signs)

Useful municipal entry points include [street centerlines](https://dev.socrata.com/foundry/data.sfgov.org/3psu-pn9h), [right-of-way polygons](https://catalog.data.gov/dataset/right-of-way-polygons), [street trees](https://dev.socrata.com/foundry/data.sfgov.org/tkzw-k3nq), and [SFMTA striping drawings](https://www.sfmta.com/Striping). Their existence is verified; completeness, permissions for each linked drawing, installed-state accuracy, and ingestion effort have not been audited.

**Where paid data can save the month**

Prioritize recent street imagery with calibrated camera poses and preferably matching mobile LiDAR, then recent true orthophotos and obliques. These can resolve facade identity, pole position, lane geometry, and recent construction. A textured city mesh is an acceleration option where it passes the same tests.

Aerometrex's published San Francisco catalog lists a 19.1 km² model at 5 cm GSD and a 5.2 km² model at 2 cm GSD, captured March–June 2021. Coverage may overlap. These are potentially useful regional assets, not evidence of a current, complete San Francisco model. Prices require a quote. [Aerometrex catalog](https://aerometrex.com/models/)

Nearmap provides a coverage tool listing available content types and planned captures. Check the actual capture footprints and dates for the pilot and remaining city. A national refresh claim cannot establish San Francisco freshness. [Nearmap coverage documentation](https://help.nearmap.com/kb/articles/241-nearmap-coverage)

Cyclomedia offers imagery and LiDAR access products, making it a candidate to investigate. San Francisco coverage, capture dates, bulk reconstruction access, and price were not established in this research. A viewer API alone is insufficient. [Cyclomedia API description](https://developer.cyclomedia.com/our-apis/street-smart/)

Mapillary offers imagery and downloadable predicted map features. Check capture dates and image coverage per facade; do not assume its predicted locations are surveyed ground truth. Its imagery licensing and map-data/API terms are distinct inputs to the source register. [Mapillary feature downloads](https://help.mapillary.com/hc/en-us/articles/4407521157138-Downloading-map-data-via-the-Mapillary-web-app), [imagery license](https://help.mapillary.com/hc/en-us/articles/115001770409-CC-BY-SA-license-for-open-data)

Before selecting a paid source, obtain a representative sample, actual capture dates, coverage polygons, positioning accuracy, camera calibration or pose information, supported bulk formats, delivery timing, and a price. The license must support the intended processing: local storage, machine analysis, deriving geometry, baking imagery into textures, and using the result in the private game. These are separate capabilities from online viewing. No purchases or vendor outreach have been performed.

Google's ordinary Map Tiles policies prohibit machine interpretation, object detection, geodata extraction, and offline use, subject to the applicable agreement. Therefore Google tiles and Street View are not assumed to be raw material for this extraction pipeline. Private use does not automatically supply missing permissions. A permitted streamed visualization would be a different architecture with different constraints. [Google Map Tiles policies](https://developers.google.com/maps/documentation/tile/policies), [Google Maps Platform terms](https://cloud.google.com/maps-platform/terms)

**Targeted visual inspection: Google Maps and Mapillary**

Street-level visual evidence is a core part of the design. Mapillary should be evaluated during initial coverage discovery, alongside paid imagery, rather than reserved for a last resort. Its original image sequences and available metadata are preferable to screenshots when permitted access supplies better resolution and camera information. Inventory detections and image observations are separate evidence types.

Google Maps and Street View can have high technical value as visual references. A newer street image may reveal a repainted facade, different windows, new storefront, changed lane configuration, or removed pole that an older scan misses. That image can update an appearance hypothesis without requiring the whole building's measured geometry to be discarded. This is a proposed evidence-fusion strategy, not an assertion that every source permits every downstream use.

Assess the specific Google product and intended use. Maps Platform/API restrictions are not identical to the consumer screenshot permissions. Google's consumer terms and Geo Guidelines permit certain viewing, annotation, and content-display uses while imposing copying and reuse restrictions. An occasional reference screenshot is a different workflow from mass collection or extracting an entire city's geometry and textures; neither private use nor the screenshot format automatically establishes permission for reconstruction. Keep Google as an explicit conditional source instead of assigning its evidence zero technical value. [Consumer terms](https://www.google.com/help/terms_maps/), [Geo Guidelines](https://about.google/brand-resource-center/products-and-services/geo-guidelines/)

For sources eligible for the intended analysis, use this bounded inspection loop:

1. Flag a concrete uncertain attribute or visible mismatch, such as bay-window count or a lane-divider change.
2. Select a small set of dated, complementary views that can resolve it. Choose by visibility, resolution, registration quality, and actual capture date, rather than provider name.
3. Use existing measured geometry to anchor image interpretation. Recover a compact parameter patch and its evidence references; do not ask the language model to regenerate the whole asset.
4. Validate the patch against another view where available, then rebuild only affected outputs. Stop when the evidence cannot resolve the remaining uncertainty.

Preserve source links or image IDs, capture dates, attribution, and known camera information. A screenshot alone often omits calibration, depth, and exact viewpoint; it can establish visible appearance more readily than centimeter-accurate position. Google provides access to other Street View dates where available and does not promise a fixed update timeline. The date the screenshot was taken is not the date the street was photographed. [Street View dates and updates](https://support.google.com/maps/answer/3093484?hl=en-GB)

**Core representation**

Use a geographic feature database as the source of truth. Game meshes are generated outputs. Store buildings, roof parts, facade patches, road surfaces, lane boundaries, markings, signs, poles, trees, water boundaries, bridge decks, and tunnel portals as separate related features.

Each feature has a stable internal ID, source aliases, spatial extent, attributes, and dependencies. Each attribute stores:

```text
value and units
source ID and original record ID
actual capture/observation date or unknown
ingestion date and source version
coordinate and vertical reference
measurement uncertainty or unknown
method and method version
evidence geometry or image crop references
status: observed | inferred | synthetic | unresolved
quality tier calibrated from pilot results
license/provenance reference
```

A single building can have observed roof geometry, observed front-wall texture, inferred window depth, and synthetic rear-wall appearance. Do not flatten this into one misleading confidence score. “Observed” means backed by an observation, not guaranteed correct.

Keep separate geometry precision, image resolution, temporal age, evidence coverage, and validation results. Source confidence fields are not automatically calibrated probabilities. A visually realistic material does not raise geometric confidence.

**The reconstruction heuristic**

For every feature attribute, choose the least expensive eligible method expected to meet its error tolerance. Eligibility first requires suitable rights, a compatible coordinate frame, an appropriate observation type, sufficient coverage/resolution, and acceptable age for the selected snapshot. Missing freshness must be reported, not silently treated as fresh.

The initial escalation order is:

1. Reuse an already validated asset if its dependencies have not changed.
2. Import an authoritative or licensed measured attribute if it passes the relevant checks.
3. Fit geometry or project textures from observations using deterministic processing.
4. Run a pretrained local vision model to identify or segment evidence, then fit it geometrically.
5. Request a bounded language-model review of ambiguous, high-impact evidence.
6. Obtain a better observation or make a targeted manual correction.
7. If evidence remains inadequate, retain an explicit unresolved or synthetic fallback. The original accuracy requirement remains unmet.

Do not treat this as an immutable ordering across all modalities. A prepared licensed mesh may cost less than reconstructing it. A better image may be both cheaper and more accurate than repeated model calls. Select using pilot measurements.

For numeric attributes in the same physical state, use robust fitting with residuals normalized by measurement uncertainty. Downweight occluded, poorly registered, or obliquely viewed evidence. Group correlated sources: OSM and an Overture record derived from that same OSM feature are one lineage, not two independent votes.

Separate hypotheses for temporally conflicting states. A newly observed taller tower is not averaged with its demolished predecessor. For categorical attributes such as sign type or lane count, preserve candidate hypotheses and choose using corroboration, age, and calibrated reliability; averaging has no physical meaning.

An initial intervention priority can be expressed as:

```text
priority(action) = expected reduction in weighted error
                   / normalized incremental cost

normalized incremental cost =
    lambda_h * person_hours
  + lambda_g * compute_hours
  + lambda_d * dollars
  + lambda_t * model_tokens
```

The lambdas convert different resources into comparable budget pressure. Set them from remaining budgets and revise using the pilot. Enforce hard budget caps in addition to this ranking. This is a scheduling heuristic, not a proven global optimum.

Weighted error includes driving impact, visible image area from representative cameras, distinctiveness of the feature, probability it is wrong, and its measured deviation from the requested tolerance. Unobserved high-impact features receive a missing-evidence task. Screen visibility controls work order and rendering detail; it never changes whether a feature qualifies as accurate.

**Processing sequence**

1. **Register and tile.** Normalize units, horizontal projection, vertical datum, and camera coordinates. Preserve source originals. Align on stable control features and hold independent controls out of the fit. Start with 500 m processing tiles and approximately 50 m context margins; pilot-adjust both. Give cross-tile objects one owner and reconcile shared borders. Keep bridges or complex interchanges as connected work units when necessary. Use a local metric frame for computation and retain the transform to global coordinates.

2. **Reconcile identities.** Match buildings using overlap, boundary distance, roof height, adjacency, addresses when available, and date. Resolve splits, merged row houses, courtyards, overhangs, and roofprint-versus-footprint differences. An address alone is insufficient. Preserve original IDs and match evidence.

3. **Reconstruct ground and buildings.** Fit bare-earth terrain from ground returns. For buildings, extract local points, reject vegetation/outliers, segment roof planes, fit a regularized roof structure, and build wall geometry constrained by supported outlines. Preserve actual non-rectangular shapes rather than forcing everything orthogonal. An established option is roofer, which reconstructs LoD2 building geometry from point clouds and roofprints. It does not reconstruct every facade detail. [roofer project](https://github.com/3DBAG/roofer)

4. **Reconstruct the driving surface.** Use the street graph to identify connected routes. Measure pavement and curb boundaries from suitable imagery or LiDAR; do not buffer every centerline by a guessed uniform width. Fit piecewise road surfaces to measured elevations, retaining hill crests, crowns, banking, intersection transitions, and grade breaks. Build a separate collision surface with stable normals. Fine collision tessellation improves numerical behavior but does not create missing survey accuracy.

5. **Handle vertical crossings explicitly.** Model bridge decks, ramps, tunnels, and the ground beneath as different surfaces with different connectivity. Never drape bridge decks onto bare-earth terrain or connect roads solely because their 2D lines intersect. Preserve deck thickness, clearance, piers, abutments, and railings to the extent observed. Large bridge structures need dedicated reconstruction or a licensed asset aligned to measurements.

6. **Recover lane semantics and markings.** Extract paint masks, crosswalks, stop lines, arrows, curb colors, rail tracks, and lane boundaries from recent images. Vectorize them and reconcile with lane topology and any verified installed plans. Parse vector drawing paths before resorting to image recognition; georegister dimensions and control points. Plans may describe proposed work. Generic road standards generate a candidate prior only when the installed geometry is unknown. Render markings as separate decals or meshes so they can be corrected without rebuilding the road.

7. **Fit facades from images.** Select views by date, resolution, angle, and occlusion. Refine camera poses against reliable geometry. Project visible wall patches, reject trees/vehicles/people, and blend consistent observations. Segment windows, doors, balconies, storefronts, cornices, and bay windows. Fit a compact architectural grammar to their observed arrangement. Use multi-view evidence for depth; single-image depth remains inferred. Model silhouette-changing features geometrically and use materials for small shallow details. Preserve unique colors and layouts instead of replacing every facade with its neighborhood archetype. Unseen courtyards and back walls remain explicitly incomplete or synthetic.

8. **Place street objects.** Triangulate repeated detections using sufficiently separated, well-registered views or match them to LiDAR. Recover position, dimensions, direction, attachment, and type. Inventory points seed searches; they do not override stronger physical observations. OCR text must be corroborated before being treated as factual. Instantiate a reusable sign or pole mesh at the measured pose, with feature-specific panels and dimensions. Account for utility wires and transit overhead wires where visible; missing observations remain gaps.

9. **Generate vegetation and water.** Fuse tree candidates with canopy geometry. Reuse tree assets with measured position, canopy size, and supported species, marking exact branch form as synthetic unless reconstructed. Infer additional vegetation from imagery and point clouds, not only street-tree records. Build shorelines, piers, seawalls, and river or bay boundaries from suitable geographic evidence. Render water at the selected datum/tide state with simulated motion; do not reinterpret missing water LiDAR returns as land. Bathymetry is a separate requirement if underwater geometry matters.

10. **Compile and validate.** Export spatial chunks, texture atlases, collision meshes, instance records, semantic metadata, and quality masks. Reuse asset geometry while preserving real placements and variations. Stream detailed nearby chunks and simpler distant representations, including a horizon treatment that does not load the entire city at full detail. Unreal World Partition is one implementation option for spatial streaming; HLOD provides aggregated distant representations. The pipeline should keep an engine-independent intermediate representation. [World Partition](https://dev.epicgames.com/documentation/en-us/unreal-engine/world-partition-in-unreal-engine), [HLOD](https://dev.epicgames.com/documentation/unreal-engine/world-partition---hierarchical-level-of-detail-in-unreal-engine?lang=en-US)

**Pseudocode**

```python
# Specification pseudocode. Helpers require implementation and pilot validation.
def build_world(area, snapshot, tolerances, budgets):
    register = audit_sources(area, snapshot)
    coverage = measure_evidence_coverage(register, area)
    observations = normalize_and_index(register)
    queue = []

    for tile in processing_tiles(area, context_margin=True):
        evidence = observations.query(tile.with_context)
        features = reconcile_feature_identities(evidence)
        features = separate_temporal_hypotheses(features, snapshot)
        for feature in features:
            if dependency_cache_is_valid(feature):
                reuse(feature)
                continue
            for attribute in required_attributes(feature):
                methods = eligible_methods(attribute, evidence, snapshot)
                method = cheapest_expected_to_pass(methods, tolerances)
                if method is None:
                    queue.append(missing_evidence_task(attribute))
                    retain_labeled_fallback(attribute)
                else:
                    result = method.run(attribute, evidence)
                    store_result_with_provenance(result)
                    queue.extend(validate_or_flag(result, tolerances))

    reconcile_tile_seams_and_crossings()
    queue.extend(independent_quality_checks())
    while queue and budgets.allow_next_action():
        action = select_expected_error_reduction_per_cost(queue, budgets)
        outcome = execute_bounded_action(action)
        record_actual_cost_and_error_change(outcome)
        invalidate_and_rebuild_affected_dependencies(outcome)
        update_queue(queue, outcome)

    assets = compile_game_assets()
    report = audit_all_coverage_accuracy_freshness_and_runtime(assets)
    return assets, report  # Budget exhaustion never implies the target was met.
```

**Token and compute discipline**

Do not send point clouds, meshes, citywide JSON, or every street image into a language model. Keep spatial operations in ordinary software and visual inference in bounded batch jobs. The language model writes code, develops reusable rules, diagnoses grouped failures, and inspects selected evidence crops. Successful generation jobs need only summary statistics and changed issue counts in chat.

Cache by source content/version, spatial extent, relevant method version, and configuration. An update to a facade image invalidates that facade and its compiled outputs, not the city's terrain. Store evidence outside the conversation and send stable IDs with small relevant excerpts. Cluster recurrent errors so one fix can repair thousands of instances. Prefer bounded retry counts and explicit escalation over unattended model debate loops.

For illustration only, 100,000 features at 2,000 tokens each would require 200 million tokens. Reviewing 1% of those features at 1,500 tokens each would require 1.5 million tokens for those reviews. The 1% rate and token sizes are hypotheses to measure, not promises; image inputs, repeated context, reasoning, retries, and pipeline-development work add usage. The actual feature count is not established here.

Track these budgets independently: active person-hours, language-model usage, paid data/software, and machine time/storage. Token-efficient processing can still be expensive in GPU hours or data licenses. Conversely, ordinary local scripts can process millions of records without per-record language-model usage.

The user's phrase “GPT max plan” is treated as an existing high-usage subscription, not a verified plan entitlement. Programmatic API use must have its own cost line: OpenAI documents API-key usage as separately billed at API rates rather than consuming included ChatGPT plan credits. No numerical monthly token allowance is assumed. [OpenAI authentication and billing documentation](https://learn.chatgpt.com/docs/auth)

Start with one capable workstation or rented worker and tiled processing. Measure peak RAM/VRAM, disk volume, throughput, and download overhead on the pilot before sizing more hardware. Storage for all facade observations, texture versions, and intermediate geometry can dominate. Worker scaling is limited by I/O, shared stages, memory, and serial integration work.

**Accuracy contract and independent checks**

The following are proposed pilot targets, not claimed source accuracy or accepted substitutes for the user's eventual complete-detail objective. Validate that reference uncertainty is materially below each tolerance; otherwise report the metric as unverified.

| Quantity | Initial target | Measurement |
|---|---|---|
| Road horizontal position | 95th-percentile error ≤ 0.25 m | Independent curb/edge control samples |
| Driving-surface elevation | 95th-percentile error ≤ 0.15 m | Independent road-elevation samples and grade profiles |
| Lane and marking position | 95th-percentile error ≤ 0.15 m | Registered independent high-resolution images or survey |
| Building wall/roof boundary | 95th-percentile error ≤ 0.50 m | Independent geometry samples and silhouette comparisons |
| Sign/pole base position | 95th-percentile error ≤ 0.30 m | Independent surveyed or multi-view reference positions |
| Lane count and crossing connectivity | No known errors on release-tested routes | Automated topology checks plus manual review |
| Facade correspondence | Report coverage and measured feature errors | Held-out images; window/door layout and material checks |
| Freshness | Publish age histogram and unknown fraction per layer | Actual capture dates, never catalog timestamps |

These tolerances are demanding and can require better data than the open baseline. A 0.25 m DEM cell or a 7.5 cm image pixel does not guarantee a corresponding positional error. Avoid estimating fine line widths from only one or two source pixels. No universal facade “accuracy percentage” is meaningful without a defined feature set and denominator.

Report coverage both by total feature counts and relevant physical measure: road length, facade surface area, building coverage, and land area. Report observed/inferred/synthetic/unresolved shares for every class. Sample random features as well as high-risk areas; make independent imagery available for testing where licenses permit. Checks against input data measure internal fit, not independent real-world truth. Human review of a sample cannot certify that all unsampled objects are exact.

For gameplay, test collision continuity, intersection transitions, bridge clearances, tunnel entrances, tile seams, rail crossings, and high-speed streaming. A proposed rendering benchmark is 1080p at 60 fps on a named target machine, with 95th-percentile frame time ≤ 16.7 ms and separate reporting of worst streaming hitches. Freeze camera height, FOV, speed range, lighting, and hardware for comparisons. Include slow driving and stationary inspection so speed does not hide defects.

**First-week experiment and month schedule**

Choose six approximately 500 m pilot tiles spanning dense towers, steep hill streets, attached residential facades, recent redevelopment, trees/park edges, and a waterfront or elevated crossing. Select exact extents after coverage discovery. Reserve separate held-out areas for final validation. Avoid using one easy downtown block to predict the whole city.

For each pilot area, measure licensed recent evidence coverage, automatic pass rate, residual error, processing time, active correction time, token usage, and resulting game performance. Record failure classes and their frequency. Sample several buildings, signs, intersections, and facades within each area. Increase the sample if variation makes citywide extrapolation unstable.

Use a stratified forecast rather than one global tile average:

```text
person_hours = fixed_pipeline_hours
             + sum(tile_count[stratum] * correction_hours[stratum])
             + final_validation_hours

compute_hours = sum(tile_count[stratum] * processing_hours[stratum])

wall_clock_days = procurement_and_delivery_delay
                + critical_path_of_development_processing_and_validation
```

Carry uncertainty intervals and explicit worst-case strata. Parallel execution reduces only parallelizable machine work; it does not remove source delivery delays or human review. Approximately 121.5 km² corresponds to 486 full-area equivalents of 500 m tiles, before boundary fragmentation and processing margins. This is a sizing calculation, not a surveyed tile count.

One planning allocation is 160 active hours across 30 calendar days:

| Days | Work | Person-hours |
|---|---|---:|
| 1–3 | Source audit, samples, snapshot definition, coordinate registration | 16 |
| 4–7 | Representative pilot and measured feasibility gate | 24 |
| 8–12 | Reusable reconstruction, facade fitting, road and export pipeline | 32 |
| 13–18 | Citywide batches and systemic error fixes | 24 |
| 19–24 | Highest-impact unresolved features and new-observation integration | 40 |
| 25–30 | Independent validation, collision/streaming fixes, packaging | 24 |

This is an aggressive experiment, especially if the person has no geospatial or game-engine experience. Procurement must finish early; the schedule includes no assumption that a custom citywide survey can be commissioned and delivered immediately.

At the end of week one, continue toward the exact month target only if adequate observations are available and the projected correction burden, delivery time, compute cost, and validation effort fit the remaining budgets. After pilot and pipeline work, 40 correction hours across roughly 486 tile equivalents averages under five minutes per tile. If ordinary tiles need tens of minutes of individual repair, the month estimate fails unless a reusable fix removes that work.

If the gate fails, publish the bottleneck and revised forecast. Possible user decisions are additional data, more time, or a separately defined lower-fidelity first milestone. Do not silently change the scope to a few polished corridors and claim completion of all San Francisco.

**Applying the same system elsewhere**

Keep reconstruction and compilation independent of source providers. Each city supplies adapters for footprint identities, terrain, road rules, imagery, licenses, dates, and coordinate systems. Map those adapters into the shared feature and evidence schema. Local architecture and sign catalogs are reusable components, with their geometry and placement still fitted to observations.

Some cities will have excellent municipal 3D data and others almost no usable street imagery. The algorithm transfers; data coverage and labor do not automatically transfer. Publish a per-city feasibility report before generating detailed assets.

The decisive experiment is whether six representative areas can be produced mostly automatically with the required observed detail. That provides a defensible estimate for San Francisco and identifies precisely which additional observations or reusable fixes are worth buying or building.
