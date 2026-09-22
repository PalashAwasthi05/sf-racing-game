from pipeline.facades import observed_material

def test_one_frontage_does_not_color_whole_building():
 p={'match':{'featureId':'any-id'},'material':'brick_dark'}
 assert observed_material([p],{'id':'any-id'}) is None

def test_explicit_whole_envelope_evidence_can_transfer():
 p={'match':{'featureId':'any-id'},'materialEvidence':{'scope':'whole-envelope','family':'brick','sourceIds':['photo']}}
 assert observed_material([p],{'id':'any-id'})=='brick_modular'

def test_material_transition_blocks_whole_envelope_transfer():
 p={'match':{'featureId':'any-id'},'materialEvidence':{'scope':'whole-envelope','family':'brick','sourceIds':['photo']},'panels':[{'material':'panel_silver'}]}
 assert observed_material([p],{'id':'any-id'}) is None
