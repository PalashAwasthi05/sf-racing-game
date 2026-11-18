# First photo-to-module pilot

This is an isolated addition to the existing reconstruction pipeline. It does
not replace or publish the existing Market Street or Pacific Heights worlds.
No third-party city assets were purchased or copied.

## Purpose

Recover repeating shallow surface detail from a material zone in a street
photograph. Existing facade geometry already supports reusable components;
blank or incorrectly parameterized surfaces are a separate input problem.

`general_world/facade_joint_grid.py` measures dark joint repetition, searches
bounded shear/convergence, and checks the fitted grid in withheld image strips.
Windows and vegetation must be excluded by the supplied surface mask. The CLI
can propose that mask from color distance; this is explicitly provisional.

The detector has no geographic identifiers, addresses or per-building branches.
Source identities, crops, hashes and mask settings remain in experiment output.

## Use

From the repository root, with numpy, scipy, Pillow and pytest installed:

```sh
PYTHONPATH=. python experiments/measure_facade_joints.py SOURCE.jpg \
  --mask SURFACE_MASK.png --output experiments/joint-output \
  --min-pitch-px 18 --max-pitch-px 140
python -m pytest test_facade_joint_grid.py -q
```

The optional crop is in original-image pixels. Search pitch is in processing
pixels after the CLI's documented maximum 1100px downsample. The report records
both frames. At most one image is decoded; no browser, model or world build runs.

Outputs: `measurement.json`, `surface-mask.png`, `joint-overlay.png`, `review.png`
and an illustrative `module-layout.png`. The last is a layout swatch, not a PBR
material or a production texture. Source photography is never used as its pixels.

## First source check

An external crop from the cached source photograph identified approximately
68px vertical panel spacing, using 11 fitted joint lines. Withheld strips supported
90% and 100% of predicted visible lines. Horizontal spacing remained unresolved.
A larger mixed-region crop failed the withheld-strip gate and was rejected.
The successful focused run took about one second and recorded 140 MiB peak RSS.

The source/mask region was selected manually. This is not automatic facade
segmentation or an end-to-end reconstructed building. A grid score and its image
strip checks do not certify material identity, accurate physical dimensions,
independent-camera agreement, exact joints or final rendered appearance.

## Integration boundary

Accepted here means **image-space repetition only**. Existing metric material
recipes require measured module dimensions and a material classification.
Do not turn this output directly into `moduleM`, copy it to another wall, infer
a missing second joint axis, or claim the driving world changed. Registration,
source surface ownership and physical scale must supply those missing facts.

The next production step is to feed a registered material-zone crop through this
measurement and the existing metric PBR recipe, then regenerate the affected
facade and compare from its source and held-out cameras. Repeat the workflow on
an adjacent facade before extending across the block.
