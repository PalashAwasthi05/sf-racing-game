"""Rebuild all scoped source-identified sign recipients with a staged support rule."""
import json,runpy,sys,types
from pathlib import Path
from pipeline import facade_grammar
def main():
    source=Path(facade_grammar.__file__).read_text()
    needle=" for sg in profile.get('signs',[]):\n  for poly,depth,thickness,material in facade_signs.parts"
    add=""" from regen import opening_lettering
 for opening_with_infill in ops:
  for poly,depth,thickness,material in opening_lettering.infill_parts(opening_with_infill):
   prism(poly,depth,thickness,material,detail)
 for sg in profile.get('signs',[]):
  if sg.get('mount') in ('opening-glass','opening-infill'):
   from regen import opening_lettering
   for poly,depth,thickness,material in opening_lettering.parts(sg,W,Hs,ops):
    prism(poly,depth,thickness,material,detail)
   continue
  for poly,depth,thickness,material in facade_signs.parts"""
    assert source.count(needle)==1
    mod=types.ModuleType('pipeline._candidate_grammar');mod.__package__='pipeline'
    exec(compile(source.replace(needle,add),str(facade_grammar.__file__),'exec'),mod.__dict__)
    # The generator imports the public draw function from the package at run time.
    for key in ('compile_wall',):
        if not hasattr(mod,key):raise ValueError('unknown facade emitter')
        setattr(facade_grammar,key,getattr(mod,key))
    path=Path(__file__).resolve().parents[1]/'generate_detailed_v14.py';sys.argv[0]=str(path);runpy.run_path(str(path),run_name='__main__')
if __name__=='__main__':main()
