"""Stage the same structural changes on unchanged generator files for a
source-camera comparison before applying them to the next complete world."""
from pathlib import Path
from regen.try_opening_lettering import facade_grammar
def stage():
    work=Path(__file__).resolve().parents[1];prod=work/'pipeline';dst=work/'regen/staged_source_windows'
    text=(prod/'bays.py').read_text()
    for start,end,fn,bnds,indent in [("                if st.get('sash') == 'double-hung':","                sill = box","bar","(ox1,oz1,ox2,oz2)",16),("            if st.get('sash') == 'double-hung':","            sill = box","chord_bar","(wx0,wz0,wx1,wz1)",12)]:
        i=text.index(start);j=text.index(end,i);space=' '*indent
        text=text[:i]+space+"from .window_panes import inner_bars\n"+space+f"for strip,near in inner_bars({bnds},st):\n"+space+f"    {fn}(strip,near,GLASS_IN_M)\n"+text[j:]
    (dst/'bays.py').write_text(text)
    text=(prod/'facade_detail.py').read_text();text=text.replace("        elif sash == 'casement' and ix1 - ix0 > .5:\n            parts = [(iy0, iy1, d0)]; B(cx, (iy0 + iy1) / 2, .07, iy1 - iy0, .09, frame, d0 + .02)\n","")
    (dst/'facade_detail.py').write_text(text)
    text=(prod/'facade_grammar.py').read_text()
    assert text.count("if kind in ['entry','shopfront'] and not (kind=='entry' and s.get('door')):")==3
    import re
    text,n=re.subn(r"(?m)^( +)if kind in \['entry','shopfront'\] and not \(kind=='entry' and s.get\('door'\)\):\n\1 for x in \[cx-.12,cx\+.12\]:block\([^\n]+\)\n","",text);assert n==3
    text=text.replace("get('mullionFractions',[.25])","get('mullionFractions',[])").replace("get('transomFractions',[.76])","get('transomFractions',[])")
    a=text.index('  # Stable blinds, only behind the glazing.');b=text.index(' # Details read from the photograph',a);text=text[:a]+text[b:]
    a=text.index(" for sg in profile.get('signs',[]):\n  for poly,depth,thickness,material in facade_signs.parts")
    text=text[:a]+""" from . import opening_lettering
 for op in ops:
  for poly,depth,thickness,material in opening_lettering.infill_parts(op):
   prism(poly,depth,thickness,material,detail)
 for sg in profile.get('signs',[]):
  if sg.get('mount') in ('opening-glass','opening-infill'):
   for poly,depth,thickness,material in opening_lettering.parts(sg,W,Hs,ops):
    prism(poly,depth,thickness,material,detail)
   continue
"""+text[text.index('  for poly,depth,thickness,material in facade_signs.parts',a):]
    (dst/'facade_grammar.py').write_text(text);(dst/'opening_lettering.py').write_bytes((work/'regen/opening_lettering.py').read_bytes())
if __name__=='__main__':stage()
