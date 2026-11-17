# Imported player car

Since 2026-11-16 the player car is an imported model, not the generated roadster (`car/build_roadster.py` still builds
the old one).

Source: "2017 Porsche 718 Boxster S" by Ddiaz Design on Sketchfab
(https://sketchfab.com/3d-models/none-4b52cfc4ce32438aa732880fae81c047), CC BY-NC-SA 4.0. The page's tags name a
commercial racing game as the mesh's origin, so the licence the uploader gives may not be theirs to give: fine for
private use, check before publishing. The FBX and textures are not in this repository.

Convert (no Blender needed; three.js FBXLoader and GLTFExporter in a headless page):
1. Unzip the model's `source/*.zip` to `<dir>/dl/boxster/`, copy `king-street/node_modules/three/{build,examples/jsm}`
   to `<dir>/fbx/three/`, and these files to `<dir>/fbx/`. Serve `<dir>` on 127.0.0.1:3011 with CORS.
2. `python -I make_gts_badge.py <dir>/dl/boxster <dir>/dl/cayman <dir>/dl/boxster/Badge_GTS.png` (only for `gtsBadge`)
   `node convert.mjs out.glb '{"wheelColor":"#3a3a3c","open":true,"gtsFront":true,"gtsDiffuser":true,"gtsBadge":1.245}'`
   `gtsBadge` (installed since 2026-11-17): the rear lettering reads "718 Boxster GTS", not "718 Boxster S". The
   script redraws that strip of the Boxster's badge sheet with the "GTS" from the Cayman's sheet; the lettering is
   1.245 times as long, so the converter widens the badge quad by the option's value, following the tail's curve.
   The real car's badge also carries "4.0", for which neither sheet has glyphs; the door-sill plates still read
   "Boxster S".
   `gtsFront` (needs `<dir>/dl/cayman/`, the "2018 Porsche 718 Cayman GTS" by the same author): cuts the Boxster's
   front bumper and headlamps out by nearest-panel classification against the Cayman's named panels and adds the
   Cayman's `pBumperF6`, `pHeadLight*` (PDLS+ four-point units) and `pBumperChassisF6`. The two bodies coincide to
   millimetres after a quarter turn and offset (.0265, -.03, 0).
   `gtsDiffuser` (needs `dl/cayman/` and `dl/spyder/`; this is what is installed): the Cayman's `pDiffuser6` and
   `pExhausts6` panels mark the diffuser zone; the Boxster's triangles in it are dropped and the Spyder's triangles in
   it are added (finned diffuser with two separate pipes).
   `gtsPipes` (older, tips only; needs `<dir>/dl/spyder/`, the "2020 Porsche 718 Spyder"): removes the Boxster's centre tailpipes and
   adds the Spyder's two separate tips. The Boxster's diffuser stays (the Spyder's is fused with its bumper), so its
   centre notch is empty.
   `open` removes the soft top and the rear and side glass (the model has the cover and hoops underneath).
3. Copy `out.glb` to `king-street/public/vehicle/roadster.glb`.

Layout written for `king-street/app/car.ts`: `body__paint` takes the game's paint; `*__src_*` parts keep the model's
textured materials; `wheel_front/rear__*` and `hub_front/rear__*` are the right-hand wheel and caliper moved to the
wheel centre (the viewer clones and mirrors them). Body is shifted +47.5 mm in x (axle midpoint at the origin) and
lifted 26 mm (the model's tyres sink below its ground plane).
