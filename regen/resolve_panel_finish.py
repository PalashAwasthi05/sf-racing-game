"""Migrate legacy untyped metal-panel keys in source profiles; explicit finish
observations and keys with existing typed descriptors are kept intact."""
import json,argparse,re
from pathlib import Path
from pipeline.panel_finish import panel_recipe
def main():
    ap=argparse.ArgumentParser();ap.add_argument('input',type=Path);ap.add_argument('output',type=Path);a=ap.parse_args();g=json.loads(a.input.read_text());changed=[]
    for p in g['profiles']:
        if not p.get('sourceIds'):continue
        def apply(q):
            if isinstance(q,list):
                for item in q:apply(item)
            elif isinstance(q,dict):
                for k,v in list(q.items()):
                    if k=='material' and isinstance(v,str) and re.fullmatch('panel_silver#[0-9a-fA-F]{6}',v) and not q.get('materialEvidence'):
                        base=panel_recipe(v[-7:],q.get('finish'),q.get('coating'))
                        if base=='panel_coated':q[k]=base+v[-7:];changed.append(p['id'])
                    elif k not in ('reading','readingProvenance','materialEvidence'):apply(v)
        apply(p)
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(g,separators=(',',':')));print('coated panel assumptions',len(changed),'across',len(set(changed)),'source profiles')
if __name__=='__main__':main()
