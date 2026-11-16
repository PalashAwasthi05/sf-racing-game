from regen.remap_readings import remap


class AffineMap:
    L = 10.

    def __call__(self, x, z):
        if isinstance(x, (list, tuple)):
            return [2 * v + 1 for v in x], [3 * v + 2 for v in z]
        return 2 * x + 1, 3 * z + 2

    def stretch(self, x, z):
        return 2., 3.


def test_recess_and_nested_back_opening_remap_once_but_depth_stays_metric():
    source = {'recesses': [{
        'x0': 1., 'x1': 3., 'z0': 2., 'z1': 5., 'depthM': 1.8,
        'frontRailing': {'kind': 'metal', 'heightM': 1., 'postPitchM': .8},
        'backOpenings': [{'x0': 1.5, 'x1': 2.5, 'z0': 2.5, 'z1': 4.5, 'kind': 'glazed_door'}],
    }]}
    mapped = remap(source, AffineMap())['recesses'][0]
    assert mapped['x0'] == 3. and mapped['x1'] == 7.
    assert mapped['z0'] == 8. and mapped['z1'] == 17.
    assert mapped['depthM'] == 1.8
    assert mapped['frontRailing']['heightM'] == 3.
    assert mapped['frontRailing']['postPitchM'] == 1.6
    opening = mapped['backOpenings'][0]
    assert opening['x0'] == 4. and opening['x1'] == 6.
    assert opening['z0'] == 9.5 and opening['z1'] == 15.5
