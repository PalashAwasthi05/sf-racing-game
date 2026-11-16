# Horizontal timber module audit r1

## Result

No profile update is supported. The stockwide candidate set contains 10 profiles typed as `timber / horizontal_courses`; all 10 have two registered rectified photos and a local source reading. **Zero of 10** produced a cross-photo vertical course pitch under the current conservative gates.

The measurement splits each independently masked wall patch into four disjoint width strips. A view needs at least three accepted strips whose periods agree within 12%; both photos must then agree on physical pitch within 12%. The pixel repeat must be at least six pixels and cover at least 2.5 cycles. This is a screening result: the existing readings do not provide semantic opening masks for these candidates, and some candidate regions retain ornate windows/trim. A positive periodicity alone would not authorize a profile change until those regions are visually confirmed as solid siding.

## Near misses and controls

- `sf_201006_0032535`: one photo had a three-strip vertical repeat near 0.162 m; the other had only two agreeing strips, so the pair failed. The façade crop is dominated by ornate window surrounds and trim. This is an explicit occlusion/trim negative control, not evidence for siding pitch.
- `sf_201006_0010666`: apparent horizontal-axis periods in the first crop ranged from about 42 to 181 px across strips, while only one strip repeated in the second crop. This demonstrates why window spacing and trim bands cannot stand in for the vertical course pitch.
- `sf_201006_0012476` had a 2.26 m apparent vertical period in one view, with no agreeing period in the other. The large repeat is an architectural feature/occlusion alias, not a material module.
- Across all candidates, the horizontal-axis measurement is retained as a negative control. No horizontal repeat is used as course pitch.

## Review files

- Stockwide measurements and per-strip outcomes: `report.json`.
- Paired registered crops with exclusion masks: `patches/`.
- Overview contact sheet: `patches/contact-sheet.jpg`.
- Reproduction script: `../../../../regen/audit_horizontal_course_modules.py`.
- Inputs: `../paired-local-lettering-v5/profiles.json`, registered sides under `../../../../pacific-heights/regen/sides-reg/`, and source readings under `../readings-r24-final/`.

The crop overview is a baseline for later semantic-mask review; it is not an approval sheet for texture changes. The script only writes diagnostics and does not modify profiles.

## Model update path if stronger evidence is collected

The existing profile contract carries verified structural scale as `materialEvidence.moduleM`. For an independently supported horizontal-course pitch, the value would be `{"z": <metres>}` alongside `family: "timber"` and `pattern: "horizontal_courses"`. The recipe adapter already validates this pattern-specific field and passes it into `observed_moduleM`. A future profile update should preserve the source image IDs and add the module only after semantic masks isolate solid siding in both photos and at least two separated wall patches per photo agree.
