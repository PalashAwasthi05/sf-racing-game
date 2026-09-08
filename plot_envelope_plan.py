import json
from PIL import Image,ImageDraw,ImageFont
r=json.load(open('work/envelope-footprints.json'));im=Image.new('RGB',(1200,1200),'#fff');d=ImageDraw.Draw(im)
def xy(p):return (600+p[0]*2.1,480-p[1]*2.1)
for b in r:
 for ring in b['rings']:
  d.polygon([xy(p) for p in ring],fill='#e4e4e4',outline='#505050',width=2)
  if b['id']=='sf_201006_0000003':
   for a,c in zip(ring,ring[1:]):
    if ((a[0]-c[0])**2+(a[1]-c[1])**2)**.5>8:
     mid=[(a[i]+c[i])/2 for i in [0,1]];d.text(xy(mid),str([round(v) for v in mid]),fill='#b00000')
 d.text(xy(b['centroid']),b['osmId'] or b['id'][-7:],fill='#006090')
for a,b in [((-190,130),(62,-120)),((-110,-145),(165,85))]:d.line([xy(a),xy(b)],fill='#538aac',width=5)
im.save('work/envelope-plan.png')
