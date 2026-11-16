"""Regression checks for square bay outlines and their cornice overhang."""

import numpy as np

from pipeline import bays


def test_square_bay_stays_rectangular_and_has_overhanging_cap():
    x0, x1, z0, z1, projection = 1.0, 4.2, 2.0, 8.0, 0.8
    shape, corners = bays.corners(x0, x1, projection, "square")
    assert shape == "square"
    assert corners == [(x0, 0.0), (x0, projection), (x1, projection), (x1, 0.0)]
    assert bays.plan(x0, x1, projection, "square").area == (x1 - x0) * projection

    meshes = []

    def point(x, z, depth):
        return np.array([x, z, depth], dtype=float)

    def put(mesh, key, material):
        meshes.append((mesh, key, material))

    bays.draw(
        point, put, "walls", "trim", {"x0": x0, "x1": x1, "z0": z0, "z1": z1,
                                          "p": projection, "shape": "square"},
        [], "brick", "stone",
    )

    # Cornice's outer cap is the only plan slab extending 16 cm past both bay ends.
    x_bounds = [mesh.bounds[:, 0] for mesh, _, _ in meshes]
    assert any(np.isclose(bounds[0], x0 - 0.16) and np.isclose(bounds[1], x1 + 0.16)
               for bounds in x_bounds)
