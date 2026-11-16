# Pier ownership source-camera validation

`profiles-three-high-overlap.json` isolates source-verified, high-overlap profiles: P1 (`sf_201006_0002472`, 1.00) and two other buildings (`sf_201006_0017402`, 1.00; `sf_201006_0023351`, 1.00). Exact registered source jobs are in `source-jobs.json`; `source-photos/` contains matching crops.

`r30-host-opening-audit.json` and `r31-raw-host-opening-audit.json` check for structural host shell blocking. `r30-group-receiver-rays.json` and `r31-raw-group-receiver-rays.json` sample the eligible group-window openings and report first-hit coverage. Group windows were expanded with `profile_apertures.audit_profile`, which calls `pipeline.facade_grammar.openings`; the copied R31 `facade_grammar.openings` implementation is unchanged from the pipeline version (the copy’s pier drawing code is the only functional divergence in that file). Every grouped opening is tagged `preserve-existing-only`; no receiver geometry is synthesized.

The ray helper only indexes each owning feature’s base shell and facade detail for speed. The separate host-opening audit applies full neighborhood and terrain checks. Results are profile-specific; no sitewide or global geometry rules were applied. Captures from the full R31 material stage will be added when its final viewer link is supplied.
