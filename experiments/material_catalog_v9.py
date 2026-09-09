"""Location independent metric procedural material families."""
from __future__ import annotations
from dataclasses import dataclass
import hashlib, json
from typing import Any, Mapping
import numpy as np
from PIL import Image

_COLORS = {"red":(.48,.12,.055),"buff":(.62,.48,.30),"gray":(.42,.43,.40),"sandstone":(.57,.41,.25),"terracotta":(.55,.20,.08),"warm_white":(.72,.70,.64),"natural":(.50,.52,.50),"galvanized":(.44,.48,.47),"aluminum":(.55,.58,.56),"copper":(.38,.16,.07),"cedar":(.38,.20,.09),"oak":(.40,.27,.14),"neutral":(.55,.63,.64),"blue":(.20,.38,.50),"charcoal":(.12,.14,.14),"black":(.05,.06,.06)}
_FAMILIES = {"masonry","plaster","concrete","metal","timber","glass","roofing"}
_VALID = {"masonry":{"kind":{"brick","stone","terracotta"},"bond":{"running","stack","English","Flemish","ashlar","coursed","random","basket"},"joint":{"flush","recessed"},"finish":{"weathered","glazed","smooth"}},"plaster":{"render":{"fine","smooth","pebbled"},"finish":{"painted","limewashed","raw"}},"concrete":{"finish":{"polished","troweled","board_formed","bush_hammered"}},"metal":{"panel":{"corrugated","standing_seam","flat"},"finish":{"polished","satin","painted","oxidized"}},"timber":{"product":{"boards","shingles"},"finish":{"oiled","raw","painted"}},"glass":{"kind":{"clear","tinted","frosted"},"finish":{"low_iron","reflective","frosted"}},"roofing":{"kind":{"slate","tile","membrane","shingles"},"finish":{"natural","glazed","granulated"}}}
_DEFAULTS = {"masonry":{"kind":"brick","bond":"running","joint":"flush","color":"red","finish":"weathered","scale_m":(.86,.276)},"plaster":{"render":"fine","color":"warm_white","finish":"painted","scale_m":(1.,1.)},"concrete":{"color":"natural","finish":"board_formed","scale_m":(1.,1.)},"metal":{"panel":"standing_seam","color":"galvanized","finish":"satin","scale_m":(.60,1.20)},"timber":{"product":"boards","color":"cedar","finish":"oiled","scale_m":(.60,2.40)},"glass":{"kind":"clear","color":"neutral","finish":"low_iron","scale_m":(1.,1.)},"roofing":{"kind":"slate","color":"charcoal","finish":"natural","scale_m":(1.20,.80)}}

@dataclass(frozen=True)
class MaterialRecipe:
    family: str; parameters: dict[str,Any]; units: str; scale_m: tuple[float,float]; texture_resolution: tuple[int,int]; recipe_hash: str; maps: dict[str,Image.Image]; metallic: float; roughness: float; resolved: bool=True
    @property
    def name(self): return self.family+":"+self.recipe_hash[:12]

class MaterialCatalog:
    def __init__(self, texture_resolution=256): self.texture_resolution=max(32,min(int(texture_resolution),1024)); self._cache={}
    def _unresolved(self,family,p=None): return MaterialRecipe(family,p or {},"m",(1.,1.),(0,0),"unresolved",{},0.,1.,False)
    def resolve(self,family: str, measured: Mapping[str,Any]|None=None):
        family=str(family).lower().strip()
        if family not in _FAMILIES: return self._unresolved(family)
        p=dict(_DEFAULTS[family]); incoming=dict(measured or {})
        for k,v in incoming.items():
            if k in p or k in ("color","scale_m","resolution","color_srgb"): p[k]=v
        for k,allowed in _VALID[family].items():
            if p.get(k) not in allowed: return self._unresolved(family,p)
        if family == "masonry" and ((p["bond"] in {"ashlar","coursed","random"} and p["kind"] != "stone") or (p["bond"] == "basket" and p["kind"] != "terracotta")):
            return self._unresolved(family,p)
        try: scale=tuple(float(x) for x in p["scale_m"])
        except (TypeError,ValueError): return self._unresolved(family,p)
        if len(scale)!=2 or min(scale)<=0: return self._unresolved(family,p)
        n=max(32,min(int(p.get("resolution",self.texture_resolution)),1024)); p["scale_m"]=scale; p["color"]=_color(p.get("color_srgb",p.get("color")))
        key=json.dumps({"family":family,"parameters":p,"resolution":n},sort_keys=True,separators=(",",":")); digest=hashlib.sha256(key.encode()).hexdigest()
        if digest in self._cache: return self._cache[digest]
        maps,metal,rough=self._synthesize(family,p,n,scale); out=MaterialRecipe(family,p,"m",scale,(n,n),digest,maps,metal,rough); self._cache[digest]=out; return out
    def _synthesize(self,family,p,n,scale):
        x=np.arange(n,dtype=np.float32)[None,:]/n*scale[0]; y=np.arange(n,dtype=np.float32)[:,None]/n*scale[1]; h=np.zeros((n,n),np.float32); rough=np.full((n,n),.72,np.float32)
        if family=="masonry": h=_masonry_height(x,y,scale,p)
        elif family=="metal":
            u=x/scale[0]; panel=p["panel"]; h=.018*np.abs(np.sin(2*np.pi*12*u)) if panel=="corrugated" else (.014*np.maximum(0,np.cos(2*np.pi*6*u)) if panel=="standing_seam" else h); rough[:]={"polished":.12,"satin":.30,"painted":.55,"oxidized":.72}[p["finish"]]
        elif family=="timber": h=.008*(np.sin(2*np.pi*y/.15)+1)/2 if p["product"]=="boards" else .012*np.maximum(0,np.sin(2*np.pi*(x/scale[0]*6+y/scale[1]*4))); rough[:]=.56 if p["finish"]=="oiled" else .78
        elif family=="roofing":
            kind=p["kind"]; h=.010*np.maximum(0,np.sin(2*np.pi*x/.25)) if kind=="tile" else (.014*np.maximum(0,np.sin(2*np.pi*(x/.30+y/.20))) if kind in {"slate","shingles"} else h); rough[:]={"glazed":.20,"natural":.72,"granulated":.86}[p["finish"]]
        elif family=="concrete": h=.004*np.sin(2*np.pi*y/.20) if p["finish"]=="board_formed" else (.002*np.sin(2*np.pi*x/.035)*np.sin(2*np.pi*y/.035) if p["finish"]=="bush_hammered" else h); rough[:]={"polished":.24,"troweled":.48,"board_formed":.76,"bush_hammered":.90}[p["finish"]]
        elif family=="plaster": h={"smooth":h,"fine":.0008*np.sin(2*np.pi*x/.012)*np.sin(2*np.pi*y/.011),"pebbled":.002*np.sin(2*np.pi*x/.025)*np.sin(2*np.pi*y/.021)}[p["render"]]; rough[:]={"smooth":.45,"fine":.68,"pebbled":.86}[p["render"]]
        elif family=="glass": h=h if p["finish"]!="frosted" else .0003*np.sin(2*np.pi*x/.01)*np.sin(2*np.pi*y/.01); rough[:]={"low_iron":.12,"reflective":.08,"frosted":.48}[p["finish"]]
        if h.shape != (n,n): h=np.broadcast_to(h,(n,n)).copy()
        base=np.array(p["color"],np.float32); color=np.clip(np.broadcast_to(base,(n,n,3))+h[...,None]*3,0,1); metallic=1. if family=="metal" and p["finish"] in {"polished","satin","oxidized"} else 0.
        return _maps(color,rough,scale,h),metallic,float(rough.mean())

def _masonry_height(x,y,scale,p):
    unit={"brick":(.215,.065),"terracotta":(.30,.15),"stone":(.60,.30)}[p["kind"]]; cols=max(1,round(scale[0]/unit[0])); rows=max(1,round(scale[1]/unit[1])); w,hh=scale[0]/cols,scale[1]/rows; row=np.floor(y/hh).astype(int); offset=np.zeros_like(y)
    if p["bond"] == "running": offset=(row%2)*w/2
    elif p["bond"] == "English": offset=(row%4==1)*w/2
    elif p["bond"]=="Flemish": offset=(row%2)*w/3
    elif p["bond"]=="basket": offset=(np.floor(x/w).astype(int)%2)*w/2
    joint=.008 if p["kind"]!="stone" else .012; edge=((np.mod(x+offset,w)<joint)|(np.mod(y,hh)<joint)).astype(np.float32)
    return (.012 if p["joint"]=="recessed" else .006)*(1-edge)
def _color(v):
    if isinstance(v,str): return _COLORS.get(v,_COLORS["gray"])
    a=np.asarray(v,dtype=float).reshape(-1)
    return tuple((a/255 if a.size==3 and a.max()>1 else a).clip(0,1)) if a.size==3 else _COLORS["gray"]
def _maps(color,rough,scale,height):
    gy,gx=np.gradient(height,scale[1]/height.shape[0],scale[0]/height.shape[1]); normal=np.dstack((np.clip(128-gx*900,0,255),np.clip(128-gy*900,0,255),np.full(height.shape,255))).astype(np.uint8)
    return {"color":Image.fromarray(np.uint8(color*255),"RGB"),"normal":Image.fromarray(normal,"RGB"),"roughness":Image.fromarray(np.uint8(rough*255),"L")}

CATALOG=MaterialCatalog()
def resolve_material(family,measured=None): return CATALOG.resolve(family,measured)
def create_recipe(family,**parameters): return CATALOG.resolve(family,parameters)
