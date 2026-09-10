"""Minimal regression for host-mask subtraction leaving an unskinned wall patch.

This is intentionally generic: it models one frontage in local (x, z) metres.
"""
from shapely.geometry import box


def missing_host_patch():
    host = box(0.0, 0.0, 10.0, 8.0)

    # replacement_mask() suppresses the complete frontage edge in XY.  The
    # envelope compiler supplies only this observed sub-rectangle.
    masked_frontage = box(0.0, 0.0, 10.0, 8.0)
    emitted_envelope = box(2.0, 2.0, 8.0, 6.0)

    remaining_base = host.difference(masked_frontage)
    emitted = remaining_base.union(emitted_envelope)
    return host.difference(emitted)


def test_replacement_mask_must_be_covered_by_emitted_skin():
    gap = missing_host_patch()
    assert gap.area == 48.0, gap.area
    assert gap.area > 0.0


if __name__ == "__main__":
    gap = missing_host_patch()
    print({"missingAreaM2": gap.area, "missingBounds": tuple(round(v, 3) for v in gap.bounds)})
