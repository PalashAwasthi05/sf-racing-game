# Registered dormer section test

This is a one-building generator run for `sf_201006_0017402`, using the compiled R31 profile input and `--roof-shell`; it did not rebuild or overwrite the full area. The new rule compares a roof reading's elevation with the profile frame (`Hs`), which is the frame used to derive `z`.

The raw pilot GLB contains 24 `stucco#a9a59a` facade-detail faces spanning source z `-21.739..-20.519`. The corrected section has no such face; its separate 224-face `#e0d1a8` cornice is unchanged. The source-camera opening-center rays now hit the existing glass after the cornice, with the mansard faces removed. Thus the source-visible dormer glazing remains backed by the original glass receiver; no new glazing was created.

The per-opening receiver audit remains `0/162` expected-receiver-first for group 1 because the projecting cornice is still the first hit for all sampled rays (144 cornice hits, 18 frame hits in the aggregate). This isolates the mansard fix but does not resolve the distinct cornice visibility issue. Raw and corrected ray chains are in `receiver-ray-audit.json` and the prior raw audit `areas/review/facade-rethink/pier-ownership-r31/source-camera-validation/r31-raw-group-receiver-rays.json`.
