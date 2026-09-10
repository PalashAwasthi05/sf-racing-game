# Surface gaps audit (v12)

The general missing-surface source is **unconditional host replacement without a coverage check**. An observed envelope mask can erase the host facade across a complete frontage while the envelope graph emits only a partial patch. The uncovered remainder becomes an actual hole.

Trace:

1. `work/history/v7/build_world.py:180-181` builds `envelope_mask` from `envelopes.replacement_mask(resolved)`. `work/pipeline/envelopes.py:150-157` makes each mask a full-height XY strip from `left-.05` to `right+.05`, for the profile's entire frontage and `replaceDepthM`.
2. `work/history/v7/build_world.py:213-215` passes that mask to the ordinary facade compiler and then emits the envelope graph separately.
3. `work/pipeline/facades.py:25-28` subtracts the mask from every footprint edge, so the host facade segment is not emitted. The midpoint guard at `:35-36` also skips any segment whose midpoint is in the mask.
4. `work/pipeline/envelopes.py:178-197` emits skin only for each resolved node (`visible = region - children`, then `skin = visible - openings`). There is no assertion that the union of node regions covers the part of the host removed by the mask. A profile node spanning only 60% of the frontage therefore leaves the other 40% absent; gaps above and below the node are possible too.

## Generic counterexample

Take a 10 m wide by 8 m high host wall. Let the replacement mask suppress the complete host frontage, and let the only resolved envelope node cover x=2..8, z=2..6. The emitted union contains only that 6x4=24 m² patch, so the missing host area is 80-24 = **56 m²** (the four surrounding strips). This is reproduced by `work/test_surface_gaps_v12.py` and runs without any named-building data.

## Correction

Treat replacement as a coverage contract: compute the host surface domain removed by each replacement mask, project/union the emitted envelope node domains in the same coordinate system, and reject or preserve the host wherever coverage is absent. The general implementation should either (a) subtract only the exact observed node union from the host, or (b) retain a continuous opaque fallback for `mask - emitted_domain`; it must not use the broad profile frontage strip as a deletion mask by itself. Add a regression that requires zero uncovered area for a fully replaced domain and preserves every uncovered remainder when a partial envelope is intentional.
