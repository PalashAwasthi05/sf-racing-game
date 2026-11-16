# Optional per-surface material observations

Each record in a raw reading's `materials` list may carry these optional typed
fields alongside its existing `family`, `hex`, and measured bounds:

```json
{
  "family": "timber",
  "hex": "#8c7051",
  "pattern": "horizontal_courses",
  "finish": "matte",
  "moduleM": {"z": 0.2}
}
```

`pattern` must describe a visible surface structure, not a guess from color.
Supported family/pattern pairs are: brick (`running_bond`, `stack_bond`), stone
or concrete (`block_grid`, `smooth_grain`), render (`plaster_grain`,
`smooth_grain`), timber (`horizontal_courses`, `vertical_boards`,
`straight_shingles`, `scalloped_shingles`), metal-panel,
terracotta, or tile (`panel_grid`, `ribbed_panel`), and glass (`smooth_grain`,
`panel_grid`). Unknown pairs are rejected.

`finish` is an observed optical class: `matte`, `satin`, `glossy`, `polished`,
`rough`, or `weathered`. It sets the recipe's roughness parameter; absent finish
leaves the current family recipe unchanged. It does not alter substrate or
pattern.

`moduleM` gives measured repeat dimensions in metres on local facade axes: `x`
is horizontal and `z` is vertical. `block_grid`, `panel_grid`, and
`ribbed_panel` require both axes. A brick bond may be observed without a module;
horizontal timber-course direction may be observed without its pitch, as may
vertical board direction without board width. Shingle patterns may also be
observed without measured modules; when supplied, both axes are required, with
`x` measuring one shingle width and `z` one exposed course height. Their
procedural tile uses an even number of courses so half-width row staggering
closes at the repeat boundary. `straight_shingles` uses rectangular lower
edges; `scalloped_shingles` has a curved lower edge and corresponding height
relief. These remain separate structures from horizontal or vertical boards.
In pattern-only cases the
procedural material keeps its family recipe's standard scale, marked
`recipe_default_unmeasured`; it is not evidence about the observed building's
physical module, and the texture selector receives no measured module. If a
module is supplied, bond patterns require both axes, horizontal courses require
only `z`, and vertical boards require only `x`. Unmeasured axes are omitted
rather than estimated. Values must be positive and finite. A module without an
explicit pattern is rejected. For `ribbed_panel`, `moduleM.z` is the observed
rib pitch as well as the vertical repeat used for the periodic tile.

Painted timber joint color is derived from the observed target paint so a
recipe's original dark seam color is not carried into lighter painted
cladding. This rule applies to typed timber patterns; masonry joint colors
remain family recipe values.

When all optional fields are absent, `regen/specs.material()` retains its
existing `recipe#hex` key and the old recipe exactly. Typed variants use a
versioned recipe-name suffix before `#hex`; this preserves recipe, color,
pattern, finish, and measured module through the existing material-name handoff.
The texture selector can reuse or recolor a scan only when its declared
structure and measured module are compatible. Otherwise it records a
structure-specific new-asset request and the current procedural recipe remains
in place. Roof color alone is not a material observation and does not enter
this surface-material contract.
