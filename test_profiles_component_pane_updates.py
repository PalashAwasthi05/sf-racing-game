import json
import os
from pathlib import Path
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parent.parent
WORK = Path(__file__).resolve().parent
SIDE_ID = 'sf_201006_0094735__s2p0'


def _fixture(tmp_path):
    source_area = WORK / 'areas/pacific-heights'
    piece = next(p for p in json.loads((source_area / 'regen/sides/sides.json').read_text())['pieces'] if p['id'] == SIDE_ID)
    reading = json.loads((source_area / f'regen/readings-v2/{SIDE_ID}.json').read_text())
    area = tmp_path / 'area'
    (area / 'regen/sides').mkdir(parents=True)
    (area / 'regen/readings-v2').mkdir(parents=True)
    (area / 'regen/sides/sides.json').write_text(json.dumps({'pieces': [piece]}))
    (area / f'regen/readings-v2/{SIDE_ID}.json').write_text(json.dumps(reading))
    base = tmp_path / 'base.json'
    base.write_text(json.dumps({'profiles': [{'id': 'regen-' + SIDE_ID}], 'method': 'fixture'}))
    comp = tmp_path / 'components.json'
    bay = reading['bays'][0]
    guard = {k: bay[k] for k in ('x0', 'x1', 'z0', 'z1', 'sash', 'lites') if k in bay}
    guard['front'] = {k: bay['front'][k] for k in ('count', 'widthM') if k in bay['front']}
    guard['side'] = {k: bay['side'][k] for k in ('widthM',) if k in bay['side']}
    update = {
        'collection': 'bays', 'index': 0, 'source': guard, 'confidence': 'high',
        'front': {'lites': [2, 1]}, 'side': {'lites': [1, 1]},
    }
    comp.write_text(json.dumps([{'side': SIDE_ID, 'paneUpdates': [update]}]))
    out = tmp_path / 'out.json'
    return area, base, comp, out, update


def _run(area, base, comp, out):
    script = WORK / 'areas/ring/profiles.py'
    return subprocess.run(
        [sys.executable, str(script), str(area), str(base), str(area / 'regen/readings-v2'), str(out),
         '--nominal', '--component-observations', str(comp)],
        cwd=WORK, text=True, capture_output=True,
    )


def test_pane_updates_apply_face_scoped_fields_after_source_guard(tmp_path):
    area, base, comp, out, _ = _fixture(tmp_path)
    result = _run(area, base, comp, out)
    assert result.returncode == 0, result.stderr
    assert '"paneOverridesApplied": 1' in result.stdout
    profile = json.loads(out.read_text())['profiles'][0]
    bay_windows = [o for o in profile['openings'] if o['style'].get('sash') == 'double-hung']
    assert len(bay_windows) == 6
    assert sum(o['style']['lites'] == [2, 1] for o in bay_windows) == 2
    assert sum(o['style']['lites'] == [1, 1] for o in bay_windows) == 4


@pytest.mark.parametrize('tamper', ['stale_source', 'geometry_field'])
def test_pane_updates_reject_stale_or_untyped_changes(tmp_path, tamper):
    area, base, comp, out, update = _fixture(tmp_path)
    if tamper == 'stale_source':
        update['source']['x0'] += 0.5
    else:
        update['front']['x0'] = 1.25
    comp.write_text(json.dumps([{'side': SIDE_ID, 'paneUpdates': [update]}]))
    result = _run(area, base, comp, out)
    assert result.returncode != 0
    assert 'stale pane source guard' in result.stderr if tamper == 'stale_source' else 'unsupported fields' in result.stderr
