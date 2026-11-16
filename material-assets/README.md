# CC0 PBR material assets

Downloaded from official ambientCG sources on 2026-09-08. Exactly three materials; color, OpenGL normal, and roughness maps only. Source JPG maps are unchanged. Full archives were used to obtain these maps, then removed to avoid redundant data. No recoloring or resizing was applied.

| Role | Material | Official approximate tile extent | Resolution | Source |
| --- | --- | --- | --- | --- |
| Red brick masonry | Bricks085 | 2.4 m horizontal × 1.2 m vertical | 1024 × 512 | https://ambientcg.com/a/Bricks085 |
| Rough asphalt | Asphalt033 | 2.5 m × 2.5 m | 1024 × 1024 | https://ambientcg.com/a/Asphalt033 |
| Light gray concrete | Concrete034 | 1.1 m × 0.55 m | 1024 × 512 | https://ambientcg.com/a/Concrete034 |

## Use and scale

Use world-space or planar UV coordinates in meters divided by the listed physical tile dimensions. Preserve the rectangular aspect ratios for brick and concrete. For a surface with dimensions W × H meters, use repetitions W / tile-width by H / tile-height. Orient brick rows horizontally on building walls.

The concrete is a smooth, lightly mottled concrete/plaster surface suitable for a generic sidewalk finish. It does not contain sidewalk slab boundaries: create expansion joints and curb thickness with geometry or a separate procedural overlay. The supplied scale describes texture detail, not sidewalk slab size.

Treat color maps as sRGB and normal/roughness maps as non-color data. These are OpenGL normal maps. Use a nonmetal material response. Keep roughness maps as supplied; tune a global scalar only as needed for scene lighting. These asset choices and material setup notes do not establish photographic correspondence with any input image.

## License and provenance

All assets are CC0 1.0 Universal under ambientCG's official declaration: https://docs.ambientcg.com/license/. Reuse, modification, and redistribution, including commercial use, are permitted without attribution. Optional credit: “Textures from ambientCG, CC0.”

CC0-1.0.txt contains the official license text, and ambientCG-license-source.html preserves the provider's declaration. Each material folder contains source-metadata.json copied from the official ambientCG API. manifest.json records exact paths, sizes, map hashes, official dimensions, and source/download URLs.

## Added 2026-11-12: detail sets for the viewer's surface finish

Six more ambientCG CC0 materials, 2K JPG, colour, OpenGL normal and roughness maps only, source maps unchanged (the user approved the download): PaintedPlaster017, Plaster001, WoodSiding008, Bark016 (0.8 m tile), Grass004 (1.4 m), Ground110 (2.1 m). `source-metadata.json` in each folder is the ambientCG API record.

`pack_detail.py` packs each into one texture for the viewer (`king-street/public/materials/<id>.png`: normal in RG, roughness in B, tone relative to the scan's mean in A). `king-street/app/surface-finish.ts` projects them in world space onto plain-coloured walls and roofs (Plaster001, Ground110), bark and lawns, multiplying the surface's own colour by the scan's relative tone, so the scan's hue is never used. WoodSiding008 and PaintedPlaster017 are packed but not assigned: siding needs a wall class from the pipeline.
