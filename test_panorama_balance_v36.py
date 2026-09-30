"""The panorama white balance reads the road below the horizon, not the capture vehicle under the camera."""
import numpy as np
from panorama_balance import road_gain, BAND


def pano(road_rgb, rig_rgb=(84, 100, 97), h=400, w=800):
    p = np.full((h, w, 3), 120, np.uint8)
    p[int(h * BAND[0]):int(h * BAND[1])] = road_rgb
    p[int(h * .74):int(h * .9)] = rig_rgb               # the roof rack and decal of a roof-mounted rig
    return p


def test_a_purple_cast_on_the_road_is_removed_and_the_rig_is_ignored():
    g = road_gain(pano((145, 131, 161)))
    corrected = np.array([145, 131, 161]) * g
    assert corrected.max() - corrected.min() < 6, corrected
    assert abs(road_gain(pano((110, 110, 110)))).round(3).tolist() == [1., 1., 1.]   # a neutral road leaves the photo alone


def test_shaded_blue_asphalt_does_not_warm_the_photo():
    p = pano((110, 110, 110)); a, b = int(400 * BAND[0]), int(400 * BAND[1])
    p[a:b, :300] = (60, 66, 80)                          # a shaded stretch: sky-lit, blue
    g = road_gain(p)
    assert np.abs(g - 1).max() < .02, g


def test_grey_photographs_do_not_overturn_a_coloured_published_reading():
    import auto_read_frontages_v35 as R
    rd = lambda rgb: ({}, {'materialEvidence': {'colorRgb': rgb}})
    beige = [rd([223, 201, 167]), rd([91, 88, 77]), rd([174, 173, 159]), rd([145, 151, 140]), rd([90, 95, 84])]
    assert R.consensus(beige, keep_first=True)[0] == 0
    brick = [rd([150, 80, 60]), rd([140, 75, 58]), rd([120, 118, 116]), rd([100, 98, 97]), rd([130, 128, 127])]
    assert R.consensus(brick, keep_first=True)[0] == 0
    wrong = [rd([150, 80, 60]), rd([60, 110, 150]), rd([55, 105, 140]), rd([65, 115, 155])]   # three agree on another colour
    assert R.consensus(wrong, keep_first=True)[0] != 0


def test_clipped_highlights_do_not_dilute_the_correction():
    p = pano((145, 131, 161)); a, b = int(400 * BAND[0]), int(400 * BAND[1])
    p[a:b, :200] = (250, 250, 250)                       # white markings at the sensor's limit
    corrected = np.array([145, 131, 161]) * road_gain(p)
    assert corrected.max() - corrected.min() < 6, corrected


def test_a_faintly_tinted_grey_group_does_not_overturn_a_coloured_reading_either():
    import auto_read_frontages_v35 as R
    rd = lambda rgb: ({}, {'materialEvidence': {'colorRgb': rgb}})
    maroon = [rd([104, 74, 62]), rd([60, 52, 59]), rd([103, 94, 107]), rd([98, 90, 90]), rd([95, 84, 99])]
    assert R.consensus(maroon, keep_first=True)[0] == 0
