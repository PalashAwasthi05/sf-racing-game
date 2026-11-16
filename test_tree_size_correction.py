import numpy as np

from regen.tree_size_correction import (
    assess_lidar_height,
    deduplicate_inventory_records,
    fallback_height_m,
    project_point,
    trunk_radius,
)


def _record(tree, ordinal, species="Acer rubrum", x=0.0, dbh="12"):
    return {
        "description": f"{tree} | Tree {ordinal}",
        "qspecies": species,
        "qsiteinfo": "Sidewalk: Curb side : Cutout",
        "position": [x, 0.0],
        "radiusM": float(dbh) * 0.0254 / 2,
        "dbh": dbh,
        "sourceId": f"source-{tree}-{ordinal}-{x}",
    }


def test_duplicate_rule_requires_matching_inventory_semantics_and_near_positions():
    a = _record("1661 Pine St", 12, species="Brisbane Box", x=0.0, dbh="7")
    duplicate = _record("1661 Pine St", 12, species="Brisbane Box", x=1.25, dbh="8")
    different_ordinal = _record("1661 Pine St", 13, species="Brisbane Box", x=0.2)
    different_site = _record("1663 Pine St", 12, species="Brisbane Box", x=0.1)
    different_species = _record("1661 Pine St", 12, species="Glossy Privet", x=0.1)

    rows, audit = deduplicate_inventory_records(
        [a, duplicate, different_ordinal, different_site, different_species]
    )

    assert len(rows) == 4
    merged = next(row for row in rows if row["sourceRecordCount"] == 2)
    assert merged["dbh"] == 7.5
    assert merged["position"] == [0.625, 0.0]
    assert next(row for row in audit if row["decision"].startswith("merged"))["maxPairDistanceM"] == 1.25


def test_proximity_alone_does_not_merge_and_incomplete_identity_stays_unresolved():
    a = _record("same address", 1, x=0.0)
    b = _record("same address", 1, x=0.4, species="Different species")
    c = {"position": [0.1, 0.0], "dbh": "10"}
    rows, audit = deduplicate_inventory_records([a, b, c])
    assert len(rows) == 3
    assert all(item["decision"] == "kept" for item in audit)


def test_radius_requires_measured_dbh_and_an_explicit_unit():
    resolved = trunk_radius({"radiusM": 0.1016, "dbh": "8"})
    assert resolved["radiusM"] is None
    assert resolved["dbhUnitMetadata"] == "missing-explicit-unit"

    unresolved = trunk_radius({"dbh": "8"})
    assert unresolved["radiusM"] is None
    assert trunk_radius({"dbh": "8"}, dbh_unit="in")["radiusM"] == 0.1016
    assert trunk_radius({"dbh": "8", "dbhUnit": "cm"}, dbh_unit="cm")["radiusM"] == 0.04


def test_wall_facing_upper_tail_is_rejected_but_distributed_tall_crown_is_accepted():
    rng = np.random.default_rng(41)
    center = np.array([0.0, 0.0])

    # 30 m facade returns dominate only the direction of the nearby wall.
    wall_side = rng.uniform(-0.9, 0.9, 240)
    wall_points = np.column_stack([
        wall_side,
        rng.uniform(1.1, 2.9, len(wall_side)),
        rng.uniform(20.0, 31.0, len(wall_side)),
    ])
    tree_points = np.column_stack([
        rng.uniform(-2.5, 2.5, 150),
        rng.uniform(-2.5, 2.5, 150),
        rng.uniform(3.0, 10.0, 150),
    ])
    contaminated = assess_lidar_height(
        np.concatenate([wall_points, tree_points]), center, [0.0, 1.0], 2.1,
        fallback_seed=26765,
    )
    assert contaminated["status"] == "rejected-wall-facing-upper-tail"
    assert contaminated["heightM"] is None
    assert contaminated["wallFacingFraction"] > 0.8
    assert contaminated["fallbackHeightM"] == fallback_height_m(26765)

    # A healthy 10.8 m crown has high returns spread across the crown, not
    # just on the wall-facing side, so it remains a LiDAR measurement.
    angles = rng.uniform(0.0, 2 * np.pi, 240)
    radii = rng.uniform(0.4, 2.9, len(angles))
    z = rng.uniform(6.0, 11.0, len(angles))
    healthy = np.column_stack([radii * np.cos(angles), radii * np.sin(angles), z])
    accepted = assess_lidar_height(
        healthy, center, [0.0, 1.0], 1.7, high_threshold_m=8.5, min_high_returns=10
    )
    assert accepted["status"] == "accepted"
    assert accepted["heightM"] > 9.0
    assert accepted["oppositeWallFraction"] is not None
    assert accepted["oppositeWallFraction"] > 0.1

    # The measured same-camera control has p95 10.82 m, so it has no returns
    # in the conservative >12 m wall-contamination tail and remains accepted.
    real_control_heights = np.r_[np.linspace(4.0, 10.82, 38), 10.82, 10.82]
    real_control = np.column_stack([
        np.cos(np.linspace(0, 2 * np.pi, len(real_control_heights), endpoint=False)),
        np.sin(np.linspace(0, 2 * np.pi, len(real_control_heights), endpoint=False)),
        real_control_heights,
    ])
    assert assess_lidar_height(real_control, center, [0, 1], 1.7)["status"] == "accepted"


def test_fallback_is_deterministic_and_projection_uses_manifest_pixels():
    assert fallback_height_m("134304") == fallback_height_m(134304)
    camera = {
        "from": [0, 0, 0], "at": [0, 1, 0],
        "width": 1000, "height": 1000, "vfovDeg": 90,
    }
    px = project_point([0, 2, 0], camera)
    assert np.allclose(px[:2], [500, 500])
    assert px[2] == 2.0
