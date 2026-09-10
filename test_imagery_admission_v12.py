from general_world.evidence import inspect_record
from general_world.policy import Policy

def test_provider_pose_without_world_registration_cannot_generate_detail():
    images={str(i):{'imageSha256':str(i)*64,'captureDate':'2026-09-01',
        'camera':{'providerPoseConsistent':True,'worldRegistrationValidated':False},'viewGroup':str(i)} for i in range(2)}
    record={'producer':{'mode':'automatic','inputHash':'fixture'},'sourceIds':['0','1'],'confidence':1.,'label':'wall'}
    assert not inspect_record(record,images,Policy(),'2026-09-09')['accepted']
    for image in images.values():image['camera']['worldRegistrationValidated']=True
    assert inspect_record(record,images,Policy(),'2026-09-09')['accepted']
