from pipeline.roof_image_measurements import derive_gable_frontage


def view(apex_y=30, y_offset=0, points=None):
    # Both measured slopes end below the crop-clipped apex, but their exact
    # intersection lies above the image's valid pixel rectangle.
    v={"imageSizePx":[240,180],"padPx":10,"pixelsPerM":20,
       "pixelSegments":[[[45,80+y_offset],[95,25+y_offset]],
                        [[145,25+y_offset],[195,80+y_offset]]],
       "uncertaintyPx":1}
    if points is not None: v["pointsM"]=points
    return v


def base(views, verified=True):
    return {"frame":{"lengthM":10,"heightRefM":8},"views":views,
            "sourcePixelsVerified":verified}


def test_crop_clipped_apex_is_derived_from_visible_slopes():
    result=derive_gable_frontage(base([view()]))
    assert result["status"] == "accepted"
    assert result["acceptedLine"]
    assert result["views"][0]["apexM"]["zM"] > 8


def test_bad_segment_endpoints_outside_crop_are_rejected():
    v=view()
    v["pixelSegments"][0][0]=[5,70]
    result=derive_gable_frontage(base([v]))
    assert result["status"] == "rejected"
    assert "outside crop" in result["views"][0]["reason"]
    assert result["acceptedLine"] is None


def test_disagreeing_views_do_not_get_averaged():
    result=derive_gable_frontage(base([view(),view(y_offset=70)]))
    assert result["status"] == "conflicting"
    assert result["acceptedLine"] is None
    assert result["agreement"]["maxResidualM"] > result["agreement"]["toleranceM"]


def test_optional_metric_points_are_only_a_consistency_check():
    fabricated=[{"xM":0,"zM":2},{"xM":5,"zM":50},{"xM":10,"zM":2}]
    result=derive_gable_frontage(base([view(points=fabricated)]))
    assert result["status"] == "conflicting"
    assert result["acceptedLine"] is None
    assert result["views"][0]["metricInconsistent"]


def test_pixel_attestation_required_even_when_geometry_is_consistent():
    result=derive_gable_frontage(base([view()], verified=False))
    assert result["status"] == "unverified-source"
    assert result["acceptedLine"] is None


def test_uncertainty_is_invariant_to_image_padding():
    import copy
    original=view()
    shifted=copy.deepcopy(original)
    shifted['padPx'] += 100
    shifted['imageSizePx']=[v+200 for v in shifted['imageSizePx']]
    shifted['pixelSegments']=[[[x+100,y+100] for x,y in seg] for seg in shifted['pixelSegments']]
    a=derive_gable_frontage(base([original]))['views'][0]
    b=derive_gable_frontage(base([shifted]))['views'][0]
    assert abs(a['heightUncertaintyM']-b['heightUncertaintyM']) < 1e-10
    # Far eave needs 35 px of extrapolation; endpoint perturbations bound it
    # at 0.2625 m, independent of where the image is padded.
    assert abs(a['heightUncertaintyM']-.2625) < 1e-10


def test_valley_cannot_be_accepted_as_a_gable():
    v=view()
    v['pixelSegments']=[[[45,25],[95,80]],[[145,80],[195,25]]]
    result=derive_gable_frontage(base([v]))
    assert result['status']=='rejected'
