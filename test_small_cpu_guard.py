import importlib.util
from pathlib import Path
from unittest.mock import patch
p=Path(__file__).parent/'viewer-bench/guard_memory.py'
spec=importlib.util.spec_from_file_location('small_guard',p);g=importlib.util.module_from_spec(spec);spec.loader.exec_module(g)

def test_small_mode_cannot_run_browser_or_full_rebuild():
    command=['python',str(p.parents[1]/'regen/sides.py'),'--max-buildings','1']
    assert g.small_cpu_batch_allowed(command,384,3584)
    assert not g.small_cpu_batch_allowed(command,385,3584)
    assert not g.small_cpu_batch_allowed(command,384,3500)
    assert not g.small_cpu_batch_allowed(command[:-1]+['2'],384,3584)
    assert not g.small_cpu_batch_allowed(['node','neighborhood-circuit.mjs'],384,3584)
    assert not g.small_cpu_batch_allowed(['python','work/generate_detailed_v14.py','--max-buildings','1'],384,3584)

def test_default_still_rejects_yellow_but_small_mode_requires_sustained_reserve():
    with patch.object(g,'system_state',return_value=(2,5000)),patch.object(g.time,'monotonic',side_effect=[0,0,2,4,6,8]),patch.object(g.time,'sleep'):
        assert g.await_headroom(4000,8)==(2,5000,False)
    with patch.object(g,'system_state',return_value=(2,5000)),patch.object(g.time,'monotonic',side_effect=[0,0,2,4,6,8]),patch.object(g.time,'sleep'):
        assert g.await_headroom(4000,8,2)==(2,5000,True)
    with patch.object(g,'system_state',return_value=(4,5000)),patch.object(g.time,'monotonic',side_effect=[0,0,2,4,6,8]),patch.object(g.time,'sleep'):
        assert g.await_headroom(4000,8,2)==(4,5000,False)
    with patch.object(g,'system_state',return_value=(2,3900)),patch.object(g.time,'monotonic',side_effect=[0,0,2,4,6,8]),patch.object(g.time,'sleep'):
        assert g.await_headroom(4000,8,2)==(2,3900,False)
