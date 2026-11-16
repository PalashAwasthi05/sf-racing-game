# Porch access evidence

Porches may carry an optional `access` object. Its fields describe only what the source crop shows; the adapter never infers rails or newel posts from porch presence.

```json
"access": {
  "status": "observed",
  "steps": {
    "present": true,
    "count": 5,
    "widthM": 1.4,
    "runM": 1.6,
    "centerOffsetM": 0.0,
    "hex": "#817d76"
  },
  "rails": {"sides": ["left", "right"], "heightM": 0.9, "hex": "#5b5a55"},
  "posts": [
    {"side": "left", "position": 0.0, "kind": "newel", "heightM": 1.0,
     "widthM": 0.22, "finial": true, "hex": "#e8e4d6"}
  ]
}
```

`status` is `observed` when access is visible, or `occluded` when the source does not support a reading. An occluded or incomplete access record becomes an unresolved profile item and draws no guessed stair or rail geometry. When status is observed, `steps.present` must explicitly say whether stairs exist. If present, `count`, `widthM`, `runM`, and `centerOffsetM` are measured values. `centerOffsetM` is relative to the porch center along the wall; `runM` extends outward from the porch front. A false `present` value records a visible porch with no stairs.

`rails.sides` contains only `left` and/or `right`, as seen from the street looking toward the facade; these rails run along the edges of the stair route. No default rails are drawn. `posts` is optional and lists observed supports individually. Its `position` is a fraction from the porch edge (`0`) to the ground end (`1`); `side` uses the same left/right convention. A `newel` is drawn as a wider capped post; `finial` requests an added rounded cap form.

When `access` is absent, existing porch behavior remains: the legacy stair flag may draw centered unrailed stairs with the prior count and width defaults. Explicit access replaces those defaults. Invalid or obscured observations are kept in `unresolved` and do not fall back to boilerplate stairs.

Stair courses form a connected solid mass. Each extends below the lowest sampled terrain in its footprint, and neighboring courses overlap vertically at their shared boundary. Posts between endpoints sit on discrete tread tops; a post at position `1` sits at sampled terrain elevation. The renderer retains the supplied porch footprint and stair center when wall tiles are clipped.
