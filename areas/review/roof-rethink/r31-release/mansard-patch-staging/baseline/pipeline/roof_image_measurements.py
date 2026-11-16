"""Validate manually marked gable lines in rectified metric photographs.

This is a numeric consistency check, not an image classifier: callers must
explicitly attest that the picked pixels follow the visible roof slopes.
"""

from itertools import product
import math


def _num(v):
    return isinstance(v, (int, float)) and math.isfinite(float(v))


def _invalid(reason, views=None, verified=False):
    return {"status": "rejected", "sourcePixelsVerified": bool(verified),
            "acceptedLine": None, "views": views or [], "unresolved": [reason]}


def _pad_xy(pad):
    if _num(pad):
        return float(pad), float(pad)
    if isinstance(pad, (list, tuple)) and len(pad) == 2 and all(_num(x) for x in pad):
        return float(pad[0]), float(pad[1])
    raise ValueError("padPx must be a number or [x,y]")


def _view_points(view, frame):
    size = view.get("imageSizePx")
    if isinstance(size, dict):
        W, H = size.get("width"), size.get("height")
    elif isinstance(size, (list, tuple)) and len(size) == 2:
        W, H = size
    else:
        raise ValueError("imageSizePx must be [width,height] or an object")
    ppm, unc = view.get("pixelsPerM"), view.get("uncertaintyPx", 0)
    if not all(_num(v) for v in (W, H, ppm, unc, frame.get("lengthM"), frame.get("heightRefM"))) or ppm <= 0 or unc < 0:
        raise ValueError("invalid image scale, uncertainty, or frame")
    px, py = _pad_xy(view.get("padPx", 0))
    segs = view.get("pixelSegments")
    if not isinstance(segs, (list, tuple)) or len(segs) != 2:
        raise ValueError("exactly two pixelSegments are required")
    clean = []
    for seg in segs:
        if not isinstance(seg, (list, tuple)) or len(seg) != 2:
            raise ValueError("each segment needs two endpoints")
        pts = []
        for p in seg:
            if not isinstance(p, (list, tuple)) or len(p) != 2 or not all(_num(c) for c in p):
                raise ValueError("segment endpoint must be [x,y]")
            x, y = map(float, p)
            if not (px <= x <= float(W)-px and py <= y <= float(H)-py):
                raise ValueError("segment endpoint outside crop rectangle")
            pts.append((x, y))
        (x0,y0),(x1,y1) = pts
        if math.hypot(x1-x0, y1-y0) < 20:
            raise ValueError("slope segment shorter than 20 pixels")
        if abs(x1-x0) < 1e-9:
            raise ValueError("vertical segment cannot define frontage slope")
        clean.append((x0,y0,x1,y1))
    def line(s):
        x0,y0,x1,y1=s
        m=(y1-y0)/(x1-x0)
        return m, y0-m*x0
    (m1,b1),(m2,b2)=map(line,clean)
    if m1*m2 >= 0:
        raise ValueError("gable slopes must have opposite signs")
    if abs(m1-m2)<1e-10:
        raise ValueError("slope segments are parallel")
    ax=(b2-b1)/(m1-m2); ay=m1*ax+b1
    ordered=sorted(clean,key=lambda s:(s[0]+s[2])/2)
    left,right=ordered
    if line(left)[0] >= 0 or line(right)[0] <= 0:
        raise ValueError("segments describe a valley rather than a gable apex")
    left_inner=max(left[0],left[2]); right_inner=min(right[0],right[2])
    # The apex can be crop-clipped vertically. Its horizontal extrapolation from
    # each segment toward the gable center must remain shorter than that segment.
    if ax < left_inner-1e-8 or ax > right_inner+1e-8:
        raise ValueError("apex does not lie between the two measured segments")
    for s in (left,right):
        span=math.hypot(s[2]-s[0],s[3]-s[1])
        if min(math.hypot(ax-s[0], ay-s[1]), math.hypot(ax-s[2], ay-s[3])) >= span:
            raise ValueError("apex extrapolation exceeds measured segment length")
    length=float(frame["lengthM"]); h=float(frame["heightRefM"])
    apex_x=(ax-px)/float(ppm)
    if not 0 <= apex_x <= length:
        raise ValueError("apex lies outside frontage")
    # Evaluate the actual left and right fitted lines at frontage limits.
    ml,bl=line(left); mr,br=line(right)
    points=[{"xM":0.0,"zM":h-(ml*px+bl-py)/float(ppm)},
            {"xM":apex_x,"zM":h-(ay-py)/float(ppm)},
            {"xM":length,"zM":h-(mr*(px+length*float(ppm))+br-py)/float(ppm)}]
    return points, clean, (px,py,float(ppm),float(unc)), (ax,ay)


def _perturb_uncertainty(clean, params, frame, nominal):
    px,py,ppm,u=params
    if u == 0: return [{"xM":0.,"zM":0.} for _ in nominal]
    # Eight endpoint coordinates, each perturbed to either interval endpoint.
    ranges=[ [float(v)-u,float(v)+u] for seg in clean for p in ((seg[0],seg[1]),(seg[2],seg[3])) for v in p ]
    values_by_point=[[] for _ in nominal]
    for values in product(*ranges):
        s=sorted([tuple(values[i:i+4]) for i in (0,4)], key=lambda q:(q[0]+q[2])/2)
        # The coordinate frame must remain identical under perturbation. Do not
        # reset image padding: that silently turns the crop border into error.
        # Pixel uncertainty may cross a crop edge, so image bounds are not a
        # reason to discard an extreme and understate its uncertainty.
        if any(abs(q[2]-q[0]) < 1e-9 for q in s):
            return [{"xM":float("inf"),"zM":float("inf")} for _ in nominal]
        lines=[((q[3]-q[1])/(q[2]-q[0]), q[1]-(q[3]-q[1])/(q[2]-q[0])*q[0]) for q in s]
        (ml,bl),(mr,br)=lines
        if ml >= 0 or mr <= 0 or abs(ml-mr)<1e-9:
            return [{"xM":float("inf"),"zM":float("inf")} for _ in nominal]
        ax=(br-bl)/(ml-mr); ay=ml*ax+bl
        length=float(frame['lengthM']); height=float(frame['heightRefM'])
        pp=[{"xM":0.,"zM":height-(ml*px+bl-py)/ppm},
            {"xM":(ax-px)/ppm,"zM":height-(ay-py)/ppm},
            {"xM":length,"zM":height-(mr*(px+length*ppm)+br-py)/ppm}]
        for i,p in enumerate(pp): values_by_point[i].append(p)
    out=[]
    for i, samples in enumerate(values_by_point):
        out.append({"xM":max((abs(p["xM"]-nominal[i]["xM"]) for p in samples),default=0.),
                    "zM":max((abs(p["zM"]-nominal[i]["zM"]) for p in samples),default=0.)})
    return out


def derive_gable_frontage(record):
    """Return derived per-view lines and a cross-view consistency decision.

    A supplied per-view ``pointsM`` is treated only as an independent check of
    the pixel-derived left-eave/apex/right-eave samples; it never replaces them.
    """
    verified=record.get("sourcePixelsVerified") is True
    frame=record.get("frame") or {}
    views=[]
    if not isinstance(record.get("views"), (list,tuple)) or not record["views"]:
        return _invalid("at least one view is required", verified=verified)
    for i,v in enumerate(record["views"]):
        try:
            pts,clean,params,apex=_view_points(v,frame)
            uncertainty=_perturb_uncertainty(clean,params,frame,pts)
            if any(not math.isfinite(q['zM']) for q in uncertainty):
                raise ValueError("pixel uncertainty does not establish stable opposing slopes")
            item={"index":i,"status":"derived","pointsM":pts,"apexM":pts[1],
                  "uncertaintyM":uncertainty,
                  "heightUncertaintyM":max((u["zM"] for u in uncertainty),default=0.)}
            supplied=v.get("pointsM")
            if supplied is not None:
                if not isinstance(supplied,(list,tuple)) or len(supplied)!=3:
                    raise ValueError("provided pointsM must contain left eave, apex, right eave")
                err=max(math.hypot(float(a["xM"])-float(b["xM"]),float(a["zM"])-float(b["zM"]))
                        for a,b in zip(pts,supplied))
                item["providedPointsResidualM"]=err
                item["metricInconsistent"]=err>max(.25,2*item["heightUncertaintyM"])
            views.append(item)
        except (ValueError,TypeError,KeyError,IndexError) as e:
            views.append({"index":i,"status":"rejected","reason":str(e)})
    good=[v for v in views if v.get("status")=="derived"]
    if len(good)!=len(views):
        return {"status":"rejected","sourcePixelsVerified":verified,"acceptedLine":None,"views":views,
                "unresolved":["one or more views failed geometric validation"]}
    uncertainty=max(v["heightUncertaintyM"] for v in good)
    metric_bad=any(v.get("metricInconsistent") for v in good)
    xs=sorted({0.,float(frame["lengthM"]),*(v["apexM"]["xM"] for v in good)})
    def zline(v,x):
        ps=v["pointsM"]
        for a,b in zip(ps,ps[1:]):
            if a["xM"]-1e-8 <= x <= b["xM"]+1e-8:
                t=(x-a["xM"])/(b["xM"]-a["xM"])
                return a["zM"]+t*(b["zM"]-a["zM"])
        return ps[0]["zM"] if x<ps[0]["xM"] else ps[-1]["zM"]
    pairs=[(good[i],good[j]) for i in range(len(good)) for j in range(i+1,len(good))]
    residual=max((max((abs(zline(a,x)-zline(b,x)) for a,b in pairs),default=0.)
                  for x in xs),default=0.)
    residual=max(residual,0.)
    conflict=metric_bad or residual>max(.25,2*uncertainty)
    result={"status":"conflicting" if conflict else ("accepted" if verified else "unverified-source"),
            "sourcePixelsVerified":verified,"views":views,
            "agreement":{"maxResidualM":residual,"heightUncertaintyM":uncertainty,
                         "toleranceM":max(.25,2*uncertainty)},
            "acceptedLine":None,"unresolved":[]}
    if conflict:
        result["unresolved"].append("pixel views or supplied metric samples disagree")
    elif not verified:
        result["unresolved"].append("sourcePixelsVerified must be true before accepting a line")
    else:
        result["acceptedLine"]=[{"xM":x,"zM":sum(zline(v,x) for v in good)/len(good)} for x in xs]
    return result
