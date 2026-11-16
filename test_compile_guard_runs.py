from copy import deepcopy
from regen.compile_guard_runs import infer_guards


def sample():
    reading={'banks':[{'x0':0.,'x1':6.,'z0':4.,'z1':6.,'columns':3,'rows':1,'openingWidthM':1.}],
             'balconies':[{'x0':0.,'x1':6.,'zM':3.6,'depthM':.1,'railingHeightM':.4,'railing':'balustrade'}]}
    proposal={'recordIndices':[0],'answer':{'probabilities':{'individual_window_guards':.97}}}
    ms=[{'image':'same-source','boundsInsideSource':True,'bank':0,'cell':i,'balcony':0,
         'measuredRectM':[c-.6,3.6,c+.6,4.], 'answers':{'visible':{'noul':.8}}}
        for i,c in enumerate([1.,3.])]
    return reading,proposal,ms


def test_repeated_panels_leave_real_gaps_and_label_unseen_repeat():
    r,p,m=sample();out,a=infer_guards(r,p,m,'same-source')
    assert len(out['balconies'])==3
    assert all(a['x1']<b['x0'] for a,b in zip(out['balconies'],out['balconies'][1:]))
    assert all(b['hasDeck'] is False for b in out['balconies'])
    assert [b['layoutEvidence']['inferredRepeat'] for b in out['balconies']]==[False,False,True]
    assert a['unresolvedPattern'] is True
    assert len(r['balconies'])==1 # immutable source


def test_notes_alone_never_rewrite_geometry():
    r,p,m=sample();assert infer_guards(r,p,[],'same-source')[0] is None


def test_other_camera_boxes_cannot_be_mixed_with_metric_reading():
    r,p,m=sample();m[1]['image']='different-source'
    assert infer_guards(r,p,m,'same-source')[0] is None


def test_real_continuous_terrace_is_preserved():
    r,p,m=sample();p['answer']['probabilities']['individual_window_guards']=.1
    assert infer_guards(r,p,m,'same-source')[0] is None


def test_misaligned_or_inconsistent_image_boxes_do_not_validate_run():
    r,p,m=sample();m[0]['measuredRectM']=[1.8,3.6,3.,4.]
    assert infer_guards(r,p,m,'same-source')[0] is None
