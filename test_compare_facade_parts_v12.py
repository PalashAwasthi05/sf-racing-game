from pathlib import Path

from PIL import Image

from compare_facade_parts_v12 import compare


def _fixture(tmp_path: Path, *, second_box=(20, 20, 40, 40), same_group=False):
    paths = []
    crops = []
    for index, group in enumerate(("view-a", "view-a" if same_group else "view-b")):
        path = tmp_path / f"crop-{index}.png"
        Image.new("RGBA", (100, 100), (255, 255, 255, 255)).save(path)
        paths.append(path)
        crops.append({
            "index": index, "path": str(path), "sourceId": f"src-{index}",
            "buildingKey": "owner:opaque", "frameIndex": 7,
            "viewGroup": group, "widthM": 10.0, "diagnosticHeightM": 10.0,
            "imageSize": [100, 100],
            "metricFrame": {"origin": [0, 0, 0], "horizontal": [1, 0, 0], "vertical": [0, 0, 1]},
        })
    boxes = [(20, 20, 40, 40), second_box]
    proposals = {"crops": [{"index": i, "sourceId": f"src-{i}", "cropPath": str(paths[i]), "proposals": [{"box": list(boxes[i]), "category": "door", "score": .9}]} for i in range(2)]}
    return {"crops": crops}, proposals


def test_perfect_metric_match_agrees(tmp_path):
    provenance, proposals = _fixture(tmp_path)
    result = compare(provenance, proposals)
    assert result["counts"] == {"agreement": 1, "disagreement": 0, "unmatched": 0, "ambiguous": 0}


def test_two_metre_shift_is_disagreement(tmp_path):
    provenance, proposals = _fixture(tmp_path, second_box=(40, 20, 60, 40))
    result = compare(provenance, proposals)
    assert result["counts"]["agreement"] == 0
    assert result["counts"]["disagreement"] == 1
    assert result["examples"][0]["centerErrorM"] == 2.0


def test_same_view_group_is_rejected(tmp_path):
    provenance, proposals = _fixture(tmp_path, same_group=True)
    result = compare(provenance, proposals)
    assert result["pairing"]["sameViewGroupRejected"] == 1
    assert result["counts"]["agreement"] == 0


def test_missing_view_group_is_rejected(tmp_path):
    provenance, proposals = _fixture(tmp_path)
    provenance["crops"][1]["viewGroup"] = None
    result = compare(provenance, proposals)
    assert result["pairing"]["sameViewGroupRejected"] == 1
    assert result["pairing"]["eligibleViewPairs"] == 0


def test_nonfinite_and_degenerate_boxes_are_rejected(tmp_path):
    provenance, proposals = _fixture(tmp_path)
    proposals["crops"][0]["proposals"] = [{"box": [0, 0, float("nan"), 2], "category": "door"}, {"box": [1, 2, 1, 4], "category": "door"}]
    result = compare(provenance, proposals)
    assert result["proposalFiltering"]["invalidBoxes"] == 2
    assert result["counts"]["unmatched"] == 1


def test_blocked_car_alpha_is_filtered_even_with_high_score(tmp_path):
    provenance, proposals = _fixture(tmp_path)
    mask = tmp_path / "masked-0.png"
    image = Image.new("RGBA", (100, 100), (255, 255, 255, 255))
    image.putalpha(0)
    image.save(mask)
    obstruction = {"crops": [{"sourceId": "src-0", "cropPath": str(provenance["crops"][0]["path"]), "outputMasked": str(mask)}]}
    result = compare(provenance, proposals, obstruction)
    assert result["proposalFiltering"]["blocked"] == 1
    assert result["counts"]["agreement"] == 0


def test_empty_required_obstruction_report_is_fail_closed(tmp_path):
    provenance, proposals = _fixture(tmp_path)
    result = compare(provenance, proposals, {"crops": []})
    assert result["proposalFiltering"]["missingOrInvalidObstructionMask"] == 2
    assert result["proposalFiltering"]["notRejectedByObstructionMask"] == 0
    assert result["counts"]["agreement"] == 0


def test_column_tie_is_ambiguous(tmp_path):
    provenance, proposals = _fixture(tmp_path)
    proposals["crops"][0]["proposals"] = [
        {"box": [20, 20, 40, 40], "category": "door", "score": .9},
        {"box": [20, 20, 40, 40], "category": "door", "score": .8},
    ]
    result = compare(provenance, proposals)
    assert result["counts"]["ambiguous"] == 1
    assert result["counts"]["disagreement"] == 1


def test_geometric_views_can_share_sequence_but_require_parallax(tmp_path):
    provenance,proposals=_fixture(tmp_path,same_group=True)
    provenance['selection']={'pairingMode':'geometric'}
    for i,r in enumerate(provenance['crops']):
        r.update(clusterId='shared-reconstruction',imageSha256=str(i)*64,cameraCenterWorldM=[i*3.,-10.,5.])
    assert compare(provenance,proposals)['counts']['agreement']==1
    provenance['crops'][1]['cameraCenterWorldM']=[.5,-10.,5.]
    assert compare(provenance,proposals)['pairing']['eligibleViewPairs']==0
    provenance['crops'][1]['cameraCenterWorldM']=[3.,-1000.,5.]
    provenance['crops'][0]['cameraCenterWorldM']=[0.,-1000.,5.]
    assert compare(provenance,proposals)['pairing']['eligibleViewPairs']==0


def test_geometric_views_reject_duplicate_images_and_different_reconstructions(tmp_path):
    provenance,proposals=_fixture(tmp_path,same_group=True)
    provenance['selection']={'pairingMode':'geometric'}
    for i,r in enumerate(provenance['crops']):
        r.update(clusterId=str(i),imageSha256=str(i)*64,cameraCenterWorldM=[i*3.,-10.,5.])
    assert compare(provenance,proposals)['pairing']['eligibleViewPairs']==0
    provenance['crops'][1]['clusterId']='0'
    provenance['crops'][1]['imageSha256']='0'*64
    assert compare(provenance,proposals)['pairing']['eligibleViewPairs']==0
