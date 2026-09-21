"""Coverage-first acquisition, then bounded uncertainty reduction. No landmark priority."""
import math

def allocate(surfaces,captures,budget):
 # A capture may resolve many surfaces. All unserved visible surfaces get their
 # first observation before a previously served surface is revisited for polish.
 state={s['key']:{**s,'attempts':s.get('attempts',0)} for s in surfaces};selected=[];remaining=list(captures)
 for _ in range(budget):
  eligible=[s for s in state.values() if s.get('visibleArea',0)>0 and not s.get('passed',False)]
  if not eligible:break
  floor=min(s['attempts'] for s in eligible);frontier={s['key'] for s in eligible if s['attempts']==floor}
  scores=[]
  for c in remaining:
   covered=frontier.intersection(c.get('covers',[]))
   if not covered:continue
   gain=sum(state[k]['visibleArea']*state[k].get('uncertainty',1)*c.get('quality',1) for k in covered)/max(c.get('cost',1),.001)
   # Geometry-derived capture order is supplied upstream, not feature ID/name.
   scores.append((gain,-c.get('routeOrder',0),c))
  if not scores:break
  c=max(scores,key=lambda x:x[:2])[2];remaining.remove(c);selected.append(c)
  for k in c.get('covers',[]):
   if k in state:state[k]['attempts']+=1
 return selected,state
