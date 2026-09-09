import copy
import numpy as np
import trimesh
from general_world.billboards import build_carrier

def test_host_unchanged_and_panel_not_a_wall():
    host={'key':'surface','status':'emitted','boundaryXZ':[[0,0],[8,0],[8,6],[0,6]],'localToWorld':np.eye(4).tolist()}
    record={'kind':'billboard','hostKey':'surface','boundaryXZ':[[1,2],[7,2],[7,4],[1,4]],'thicknessM':.08,'standOffM':.15,'localToWorld':np.eye(4).tolist(),'supportAnchorsXZ':[[2,2],[6,2]]}
    original=copy.deepcopy(host);m,r=build_carrier(record,host)
    assert host==original and not r['cutsFacade'] and not r['establishesHiddenOpenings']
    assert m.is_volume and m.bounds[1,1]<0 and not r['artworkResolved']
    for bad in ({**record,'supportAnchorsXZ':[[20,2],[6,2]]},{**record,'hostKey':'absent'},{**record,'standOffM':float('nan')}):
        try:build_carrier(bad,host)
        except ValueError:pass
        else:raise AssertionError('Invalid billboard accepted')
    obstacle=trimesh.creation.box([2,2,2]);obstacle.apply_translation([4,0,3])
    try:build_carrier(record,host,[obstacle])
    except ValueError:pass
    else:raise AssertionError('Intersecting billboard accepted')

if __name__=='__main__':test_host_unchanged_and_panel_not_a_wall();print('Billboard ownership, support and collision tests passed')
