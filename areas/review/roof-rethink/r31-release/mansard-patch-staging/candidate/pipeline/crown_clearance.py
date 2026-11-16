"""Bound inferred street-facing foliage by the actual traffic surface in XYZ.
Measured canopy tops may overhang streets; only low synthetic twigs and leaf
clusters are trimmed. A mapped trunk isn't relocated by this appearance rule.
"""
import numpy as np
from shapely import contains_xy,prepare
from shapely.ops import unary_union


class CrownClearance:
    def __init__(self,plan,elev,vehicle_m=3.6,cluster_m=.9):
        self.elev=elev;self.floor=vehicle_m;self.cluster=cluster_m
        self.travel=unary_union([plan[k] for k in ('road','cycle','crossings') if plan.get(k) is not None]);self.margin=self.travel.buffer(cluster_m)
        prepare(self.travel);prepare(self.margin)
        self.card_count=0;self.limb_count=0

    def allows(self, xyz, cluster=True):
        a=np.atleast_2d(xyz);zone=self.margin if cluster else self.travel
        near=contains_xy(zone,a[:,0],a[:,1]);ok=np.ones(len(a),bool)
        if near.any():
            q=a[near];ok[near]=q[:,2]-self.elev(q[:,0],q[:,1])>self.floor+(self.cluster if cluster else .15)
        return ok

    def leaves(self,mesh):
        # Broadleaf generators emit independent four-triangle fans before the
        # export flattens vertices. Remove complete cards, never single faces.
        tri=np.asarray(mesh.faces)
        if len(tri)%4:return mesh
        fans=tri.reshape(-1,4,3)
        if not np.all(fans[:,:,2]==fans[:,0:1,2]):return mesh
        ok=self.allows(np.asarray(mesh.vertices)[fans[:,0,2]])
        if ok.all():return mesh
        self.card_count+=int((~ok).sum())
        return mesh.submesh([np.flatnonzero(np.repeat(ok,4))],append=True,repair=False) if ok.any() else None

    def limb(self,a,b):
        a=np.asarray(a,float);b=np.asarray(b,float);length=np.linalg.norm(b[:2]-a[:2]);
        if length<.10:return b
        t=np.linspace(0,1,max(3,int(np.ceil(length/.18))+1));ok=self.allows(a+t[:,None]*(b-a),False)
        if ok.all():return b
        self.limb_count+=1;first=np.flatnonzero(~ok)[0]
        return None if first<2 else a+t[first-1]*(b-a)
