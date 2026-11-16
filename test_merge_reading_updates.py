import json
import pytest
from regen.merge_reading_updates import merge_value, merge_directories, MergeConflict


def test_concurrent_aperture_and_recess_readings_preserve_both_updates():
    base = {'id': 'source-side', 'banks': [{'head': 'arch', 'widthM': 1.}], 'steps': {'count': 0}}
    first = {**base, 'banks': [{'head': 'pointed', 'widthM': 1.}]}
    second = {**base, 'recesses': [{'depthM': 2.}]}
    merged = merge_value(base, [first, second])
    assert merged['banks'][0]['head'] == 'pointed'
    assert merged['recesses'] == [{'depthM': 2.}]
    assert merged['steps']['count'] == 0


def test_conflicting_observations_are_not_resolved_by_branch_order():
    with pytest.raises(MergeConflict, match=r'\$\.banks\[0\].head'):
        merge_value({'banks': [{'head': 'arch'}]}, [
            {'banks': [{'head': 'flat'}]}, {'banks': [{'head': 'pointed'}]}])
    with pytest.raises(MergeConflict):
        merge_value({'openings': []}, [{'openings': [{'x': 1}]}, {'openings': [{'x': 2}]}])


def test_partial_branch_keeps_untouched_readings_and_ignores_reports(tmp_path):
    base, branch, output = [tmp_path / s for s in ('base', 'branch', 'out')]
    base.mkdir(); branch.mkdir()
    for identity in ('a', 'b'):
        (base / f'{identity}.json').write_text(json.dumps({'id': identity, 'steps': {'count': 0}}))
    (branch / 'a.json').write_text(json.dumps({'id': 'a', 'steps': {'count': 0}, 'recesses': []}))
    (branch / 'report.json').write_text(json.dumps({'reviewed': 2}))
    report = merge_directories(base, [branch], output)
    assert report['count'] == 2 and report['changed'] == 1
    assert json.loads((output / 'b.json').read_text())['steps']['count'] == 0
    assert all(len(s['sha256']) == 64 for r in report['readings'] for s in r['sources'])


def test_no_partial_output_on_conflicting_new_reading(tmp_path):
    base, a, b = [tmp_path / x for x in ('base', 'a', 'b')]
    for directory in (base, a, b): directory.mkdir()
    (a / 'new.json').write_text(json.dumps({'id': 'new', 'head': 'flat'}))
    (b / 'new.json').write_text(json.dumps({'id': 'new', 'head': 'arch'}))
    with pytest.raises(MergeConflict): merge_directories(base, [a, b], tmp_path/'out')
    assert not (tmp_path/'out').exists()
