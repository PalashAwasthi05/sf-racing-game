"""Experimental, masked high-frequency source-photo projection onto host walls.

No profile or production material is changed. The adapter only edits a copy of
the supplied GLB, only on planar host-wall triangles joined to the exact
profile/material record. Albedo low frequencies are removed; the source swatch
and existing texture remain the color/structure base. Absolute projection
registration is not independently verified.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import trimesh
from PIL import Image
from scipy.ndimage import distance_transform_edt, gaussian_filter
from trimesh.visual.texture import TextureVisuals


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def srgb_to_linear(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32) / 255.
    return np.where(x <= .04045, x / 12.92, ((x + .055) / 1.055) ** 2.4)


def linear_to_srgb(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, 0., 1.)
    y = np.where(x <= .0031308, 12.92 * x, 1.055 * np.power(x, 1 / 2.4) - .055)
    return np.uint8(np.clip(np.rint(y * 255.), 0, 255))


def wall_patch_coordinates(vertices: np.ndarray, normal: np.ndarray, frame_u: list[float],
                           ground_z: float, zone: dict, ppm: float) -> np.ndarray:
    """Map source-frame XYZ vertices to (pixel-x,pixel-y) in the zone patch."""
    n = np.asarray(normal, dtype=float)[:2]
    n /= np.linalg.norm(n)
    t = np.array([-n[1], n[0]])
    u = np.asarray(vertices, dtype=float)[:, :2] @ t
    z = np.asarray(vertices, dtype=float)[:, 2] - float(ground_z)
    px = (u - float(frame_u[0]) - float(zone["x0"])) * float(ppm)
    py = (float(zone["z1"]) - z) * float(ppm)
    return np.column_stack([px, py])


def select_host_faces(mesh: trimesh.Trimesh, *, normal: list[float], plane_v: float,
                      frame_u: list[float], ground_z: float, zone: dict,
                      plane_tolerance_m: float = .04, normal_dot_min: float = .94,
                      edge_tolerance_m: float = .02) -> tuple[np.ndarray, dict[str, int]]:
    """Select only outward-facing triangles on the source profile's wall plane."""
    tri = np.asarray(mesh.vertices, dtype=float)[mesh.faces]
    n = np.asarray(normal, dtype=float)[:2]
    n /= np.linalg.norm(n)
    t = np.array([-n[1], n[0]])
    face = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    lens = np.linalg.norm(face, axis=1)
    face = face / np.maximum(lens[:, None], 1e-12)
    normal_dot = face @ np.r_[n, 0.]
    plane = tri[:, :, :2] @ n
    along = tri[:, :, :2] @ t - float(frame_u[0])
    height = tri[:, :, 2] - float(ground_z)
    in_plane = np.all(np.abs(plane - float(plane_v)) <= plane_tolerance_m, axis=1)
    in_zone = (np.all((along >= float(zone["x0"]) - edge_tolerance_m) &
                      (along <= float(zone["x1"]) + edge_tolerance_m), axis=1) &
               np.all((height >= float(zone["z0"]) - edge_tolerance_m) &
                      (height <= float(zone["z1"]) + edge_tolerance_m), axis=1))
    selected = (normal_dot >= normal_dot_min) & in_plane & in_zone & (lens > 1e-8)
    return selected, {"totalFaces": int(len(mesh.faces)), "normalFaces": int(np.sum(normal_dot >= normal_dot_min)),
                      "planeFaces": int(np.sum(in_plane)), "zoneFaces": int(np.sum(in_zone)),
                      "selectedFaces": int(np.sum(selected))}


def _bilinear_repeat(image: np.ndarray, uv: np.ndarray) -> np.ndarray:
    """Sample a repeating material map at normalized coordinates, preserving sRGB bytes."""
    tex = np.asarray(image.convert("RGB") if isinstance(image, Image.Image) else image, dtype=np.float32)
    h, w = tex.shape[:2]
    u = (uv[:, 0] % 1.) * w - .5
    v = (uv[:, 1] % 1.) * h - .5
    x0, y0 = np.floor(u).astype(int), np.floor(v).astype(int)
    fx, fy = u - x0, v - y0
    x1, y1 = (x0 + 1) % w, (y0 + 1) % h
    x0, y0 = x0 % w, y0 % h
    a = tex[y0, x0] * (1 - fx[:, None]) + tex[y0, x1] * fx[:, None]
    b = tex[y1, x0] * (1 - fx[:, None]) + tex[y1, x1] * fx[:, None]
    return a * (1 - fy[:, None]) + b * fy[:, None]


def _prefilter_for_uv_footprint(image: Image.Image, uv_per_pixel: np.ndarray) -> Image.Image:
    """Low-pass a repeating map before reducing its UV frequency to atlas pixels.

    Material UVs can change by several texture texels in one source-photo pixel.
    Point/bilinear sampling then aliases repeated courses into broad bands. This
    separable periodic Gaussian approximates the pixel-footprint integration for
    the common axis-aligned facade mapping; it is deliberately conservative.
    """
    tex = np.asarray(image.convert("RGB"), dtype=np.float32)
    h, w = tex.shape[:2]
    du = np.asarray(uv_per_pixel, dtype=float)
    # Covariance of the UV footprint projected onto the texture's U/V axes.
    sigma_x = np.sqrt(du[0, 0] ** 2 + du[0, 1] ** 2) * w / np.sqrt(12.)
    sigma_y = np.sqrt(du[1, 0] ** 2 + du[1, 1] ** 2) * h / np.sqrt(12.)
    if max(sigma_x, sigma_y) < .35:
        return image.convert("RGB")
    filtered = gaussian_filter(tex, sigma=(sigma_y, sigma_x, 0), mode=("wrap", "wrap", "nearest"))
    return Image.fromarray(np.uint8(np.clip(np.rint(filtered), 0, 255)), "RGB")


def resample_map_to_source_frame(image: Image.Image, uv: np.ndarray, shape: tuple[int, int],
                                 uv_per_pixel: np.ndarray | None = None) -> Image.Image:
    """Reproject one material channel with footprint filtering onto the shared atlas."""
    h, w = shape
    filtered = _prefilter_for_uv_footprint(image, uv_per_pixel) if uv_per_pixel is not None else image.convert("RGB")
    mapped = _bilinear_repeat(filtered, uv).reshape(h, w, 3)
    return Image.fromarray(np.uint8(np.clip(np.rint(mapped), 0, 255)), "RGB")


def build_detail_atlas(source_rgb: np.ndarray, valid_mask: np.ndarray, base_map: Image.Image,
                       uv_per_patch_pixel: np.ndarray, *, strength: float = .42,
                       feather_px: float = 3., low_frequency_sigma_px: float = 4.) -> tuple[Image.Image, dict[str, float]]:
    """Multiply masked, low-frequency-removed luminance into the original base texture."""
    rgb = np.asarray(source_rgb, dtype=np.uint8)[..., :3]
    valid = np.asarray(valid_mask, dtype=bool)
    if rgb.shape[:2] != valid.shape or len(uv_per_patch_pixel) != rgb.shape[0] * rgb.shape[1]:
        raise ValueError("source patch, mask, and UV atlas dimensions disagree")
    h, w = valid.shape
    yy, xx = np.mgrid[:h, :w]
    base_srgb = _bilinear_repeat(base_map, uv_per_patch_pixel)
    base_linear = srgb_to_linear(base_srgb)
    photo = srgb_to_linear(rgb)
    luminance = np.maximum(photo @ np.array([.2126, .7152, .0722]), 1e-5)
    log_luma = np.log(luminance)
    low = gaussian_filter(log_luma, sigma=low_frequency_sigma_px, mode="reflect")
    high = np.clip(log_luma - low, -.24, .24)
    distance = distance_transform_edt(valid)
    alpha = np.clip(distance / max(feather_px, 1e-3), 0., 1.) * valid
    gain = np.exp(high * float(strength) * alpha).reshape(-1, 1)
    atlas = linear_to_srgb(base_linear * gain)
    stats = {"validFraction": float(valid.mean()), "validPixels": int(valid.sum()),
             "strength": float(strength), "detailLogLumaP05": float(np.percentile(high[valid], 5)) if valid.any() else 0.,
             "detailLogLumaP95": float(np.percentile(high[valid], 95)) if valid.any() else 0.,
             "gainP05": float(np.percentile(gain, 5)), "gainP95": float(np.percentile(gain, 95)),
             "lowFrequencySigmaPx": float(low_frequency_sigma_px)}
    return Image.fromarray(atlas.reshape(h, w, 3), "RGB"), stats


def _fit_old_uv(mesh: trimesh.Trimesh, face_mask: np.ndarray, pixel_xy: np.ndarray) -> tuple[np.ndarray, float]:
    faces = mesh.faces[face_mask]
    ids = faces.reshape(-1)
    uv = np.asarray(mesh.visual.uv, dtype=float)[ids]
    xy = np.asarray(pixel_xy, dtype=float)[ids]
    design = np.column_stack([xy, np.ones(len(xy))])
    coeff = np.linalg.lstsq(design, uv, rcond=None)[0]
    residual = np.linalg.norm(design @ coeff - uv, axis=1)
    return coeff, float(np.percentile(residual, 95))


def _split_mesh(mesh: trimesh.Trimesh, selected: np.ndarray, new_uv_for_vertices: np.ndarray,
                material: Any, suffix: str) -> list[tuple[str, trimesh.Trimesh]]:
    out = []
    for name, face_mask in (("unchanged", ~selected), (suffix, selected)):
        if not face_mask.any():
            continue
        corners = mesh.faces[face_mask].reshape(-1)
        vertices = np.asarray(mesh.vertices)[corners].copy()
        uv = (new_uv_for_vertices[corners].copy() if name == suffix else
              np.asarray(mesh.visual.uv)[corners].copy())
        faces = np.arange(len(vertices), dtype=np.int64).reshape(-1, 3)
        part = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
        part.visual = TextureVisuals(uv=uv, material=copy.deepcopy(material) if name == suffix else mesh.visual.material)
        part.metadata = dict(mesh.metadata)
        out.append((name, part))
    return out


def _resolve_side(reading_id: str, side_paths: list[Path]) -> dict | None:
    for path in side_paths:
        data = json.loads(path.read_text())
        for piece in data.get("pieces", []):
            if piece.get("id") == reading_id:
                return piece
    return None


def project_records(scene_path: Path, profiles_path: Path, side_paths: list[Path], report_path: Path,
                    patch_dir: Path, output_dir: Path, *, strength: float = .42) -> dict[str, Any]:
    scene = trimesh.load(scene_path, force="scene", process=False)
    profiles = json.loads(profiles_path.read_text()).get("profiles", [])
    by_profile = {p.get("id"): p for p in profiles}
    report = json.loads(report_path.read_text())
    output_dir.mkdir(parents=True, exist_ok=True)
    records = []
    changed = []
    used_geometry = set()
    for record in report.get("records", []):
        rid, family = record.get("readingId"), record.get("family")
        profile = by_profile.get("regen-" + str(rid))
        side = _resolve_side(str(rid), side_paths)
        entry: dict[str, Any] = {"readingId": rid, "family": family, "acceptedForExperiment": False}
        if not profile or not side or len(side.get("views", [])) < 2:
            entry["reason"] = "exact profile and two-view side record required"
            records.append(entry); continue
        if not np.allclose(profile.get("normal", []), side.get("normal", []), atol=.002) or \
                not np.allclose(profile.get("frameU", []), side.get("frameU", []), atol=.03) or \
                abs(float(profile.get("planeV", 1e9)) - float(side.get("planeV", -1e9))) > .03:
            entry["reason"] = "profile/source crop frame mismatch"
            records.append(entry); continue
        view = record.get("views", [{}])[0]
        if str(view.get("imageId")) != str(side["views"][0].get("image")):
            entry["reason"] = "appearance patch source does not match crop catalog first view"
            records.append(entry); continue
        zone = record.get("zone") or {}
        ppm = float(side["views"][0]["ppm"])
        stem = f"{rid}__{family}"
        raw_path = patch_dir / f"{stem}__source-a-raw.png"
        mask_path = patch_dir / f"{stem}__valid-mask.png"
        if not raw_path.exists() or not mask_path.exists():
            entry["reason"] = "source patch or exact wall-only mask is missing"
            records.append(entry); continue
        source_rgb = np.asarray(Image.open(raw_path).convert("RGB"))
        mask = np.asarray(Image.open(mask_path).convert("L")) > 0
        if source_rgb.shape[:2] != mask.shape or abs(source_rgb.shape[1] - (zone["x1"] - zone["x0"]) * ppm) > 3 or \
                abs(source_rgb.shape[0] - (zone["z1"] - zone["z0"]) * ppm) > 3:
            entry["reason"] = "source patch dimensions disagree with metric crop"
            records.append(entry); continue

        candidate_names = [name for name in scene.geometry
                           if name.startswith(f"buildings__{profile['match']['partId']}__observed_{profile['id']}___")
                           and scene.geometry[name].visual.material.name == profile.get("material")]
        entry["profileId"] = profile["id"]
        entry["material"] = profile.get("material")
        entry["candidateGeometryCount"] = len(candidate_names)
        entry["pixelsPerMetre"] = ppm
        entry["groundZ"] = float(side["groundZ"])
        entry["sourcePatch"] = {"path": str(raw_path), "sha256": sha256(raw_path)}
        entry["validMask"] = {"path": str(mask_path), "sha256": sha256(mask_path)}
        entry["sourceIds"] = profile.get("sourceIds", [])
        entry["sourceImages"] = [v.get("imageId") for v in record.get("views", [])[:2]]
        entry["registrationStatus"] = {"absolutePositionValidated": False,
                                       "independentHoldoutAlignmentAccepted": bool((record.get("alignment") or {}).get("accepted")),
                                       "physicalPlaneValidated": False}
        patch_changes = []
        for name in candidate_names:
            if name in used_geometry:
                continue
            mesh = scene.geometry[name]
            if not isinstance(mesh.visual, TextureVisuals) or mesh.visual.uv is None:
                continue
            base_map = mesh.visual.material.baseColorTexture
            if not isinstance(base_map, Image.Image):
                continue
            selected, counts = select_host_faces(mesh, normal=profile["normal"], plane_v=profile["planeV"],
                                                 frame_u=profile["frameU"], ground_z=side["groundZ"], zone=zone)
            if counts["selectedFaces"] < 8:
                continue
            coords = wall_patch_coordinates(mesh.vertices, profile["normal"], profile["frameU"],
                                            side["groundZ"], zone, ppm)
            old_uv_coeff, uv_p95 = _fit_old_uv(mesh, selected, coords)
            if uv_p95 > .01:
                continue
            h, w = mask.shape
            yy, xx = np.mgrid[:h, :w]
            px = np.column_stack([xx.ravel(), yy.ravel(), np.ones(h * w)])
            uv_atlas = px @ old_uv_coeff
            uv_per_pixel = old_uv_coeff[:2, :].T
            tex_w, tex_h = base_map.size
            footprint_u = float(np.hypot(*uv_per_pixel[0]) * tex_w)
            footprint_v = float(np.hypot(*uv_per_pixel[1]) * tex_h)
            filtered_base = _prefilter_for_uv_footprint(base_map, uv_per_pixel)
            before_atlas = Image.fromarray(np.uint8(np.clip(_bilinear_repeat(filtered_base, uv_atlas).reshape(h, w, 3), 0, 255)), "RGB")
            atlas, detail_stats = build_detail_atlas(source_rgb, mask, filtered_base, uv_atlas, strength=strength)
            # UVs map the finite wall source frame to an atlas; invalid-mask pixels
            # contain the original material sampled at their original metric UV.
            new_uv = coords.copy()
            new_uv[:, 0] = new_uv[:, 0] / max(w - 1, 1)
            new_uv[:, 1] = new_uv[:, 1] / max(h - 1, 1)
            new_material = copy.deepcopy(mesh.visual.material)
            new_material.name = f"{new_material.name}__sourceDetail_{rid}"
            new_material.baseColorTexture = atlas
            # GLTF PBR maps all use the mesh's same UV set. Remap every existing
            # channel into this source-frame atlas; otherwise normal/roughness
            # detail would inherit the new color UVs and stretch across the wall.
            channel_maps = {"normalTexture": "mapped-normal.png",
                            "metallicRoughnessTexture": "mapped-metallic-roughness.png",
                            "occlusionTexture": "mapped-occlusion.png",
                            "emissiveTexture": "mapped-emissive.png"}
            mapped_channels = {}
            for attr, filename in channel_maps.items():
                old_map = getattr(mesh.visual.material, attr, None)
                if isinstance(old_map, Image.Image):
                    mapped = resample_map_to_source_frame(old_map, uv_atlas, (h, w), uv_per_pixel)
                    setattr(new_material, attr, mapped)
                    mapped_channels[attr] = filename
            parts = _split_mesh(mesh, selected, new_uv, new_material, "source-detail")
            diagnostics = output_dir / "diagnostics"
            diagnostics.mkdir(parents=True, exist_ok=True)
            before_path = diagnostics / f"{rid}__mapped-existing.png"
            after_path = diagnostics / f"{rid}__masked-photo-detail.png"
            before_atlas.save(before_path)
            atlas.save(after_path)
            channel_diagnostics = {}
            for attr, filename in channel_maps.items():
                mapped = getattr(new_material, attr, None)
                if attr in mapped_channels and isinstance(mapped, Image.Image):
                    channel_path = diagnostics / f"{rid}__{filename}"
                    mapped.save(channel_path)
                    channel_diagnostics[attr] = {"path": str(channel_path), "sha256": sha256(channel_path)}
            transform, _ = scene.graph.get(name)
            metadata = dict(mesh.metadata)
            scene.delete_geometry(name)
            for suffix, part in parts:
                node = name + "__" + suffix
                scene.add_geometry(part, node_name=node, geom_name=node,
                                   transform=transform, metadata=metadata)
            used_geometry.add(name)
            patch_changes.append({"geometry": name, "selected": counts,
                                  "uvFitP95Cycles": uv_p95, "sourceMaskCoverage": float(mask.mean()),
                                  "detail": detail_stats, "atlasSize": [w, h],
                                  "originalMapFootprintTexelsPerAtlasPixel": {"u": footprint_u, "v": footprint_v},
                                  "mapSampling": "periodic Gaussian footprint prefilter before atlas resampling",
                                  "beforeAtlas": str(before_path), "afterAtlas": str(after_path),
                                  "uvChannelsRemappedTogether": ["baseColorTexture", *mapped_channels.keys()],
                                  "channelAtlases": channel_diagnostics,
                                  "beforeAtlasSha256": sha256(before_path), "afterAtlasSha256": sha256(after_path)})
        entry["geometryChanges"] = patch_changes
        entry["acceptedForExperiment"] = bool(patch_changes)
        if not patch_changes:
            entry["reason"] = "no confident planar host triangles with a reliable UV-to-source mapping"
        else:
            for change in patch_changes:
                changed.append({"readingId": rid, **change})
        records.append(entry)

    glb_out = output_dir / "source-detail-experiment.glb"
    scene.export(glb_out)
    manifest = {"schema": "source-photo-wall-detail-experiment-r3-antialiased-map-consistent", "sceneInput": str(scene_path),
                "sceneSha256": sha256(scene_path), "profilesInput": str(profiles_path),
                "profilesSha256": sha256(profiles_path), "appearanceReport": str(report_path),
                "appearanceReportSha256": sha256(report_path), "patchDirectory": str(patch_dir),
                "sideCatalogs": [{"path": str(p), "sha256": sha256(p)} for p in side_paths],
                "outputGlb": str(glb_out), "outputGlbSha256": sha256(glb_out),
                "productionIntegration": False, "normalMapOrDepthInferred": False,
                "absoluteGeometryRegistration": "uncertain; exact profile/source frame correspondence is not independently validated",
                "records": records, "changedGeometry": changed,
                "changedMeshCount": len(changed)}
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", type=Path, required=True)
    parser.add_argument("--profiles", type=Path, required=True)
    parser.add_argument("--sides", type=Path, action="append", required=True,
                        help="registered side/crop catalog shard; repeat as needed")
    parser.add_argument("--appearance-report", type=Path, required=True)
    parser.add_argument("--patch-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--strength", type=float, default=.42)
    args = parser.parse_args()
    result = project_records(args.scene, args.profiles, args.sides, args.appearance_report,
                             args.patch_dir, args.output, strength=args.strength)
    print(json.dumps({k: result[k] for k in ("outputGlb", "changedMeshCount", "records")}, indent=2))


if __name__ == "__main__":
    main()
