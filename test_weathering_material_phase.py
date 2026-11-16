"""Guard the material shader against facet-normal-based weathering resets."""

from pathlib import Path


ROOT = Path(__file__).parent


def test_weathering_overlay_uses_continuous_world_noise_not_facet_tangents():
    source = (ROOT / "king-street/app/weathering.ts").read_text()
    tint = source.split("vec3 wxTint(vec3 albedo){", 1)[1].split(
        "// Surface character by kind", 1
    )[0]

    assert "wxLarge=wxFbm3(P*.045)" in tint
    assert "wxNoise3(P*1.7)" in tint
    assert "dot(P.xz,T.xz)" not in tint
    assert "vec3 T=normalize" not in tint


def test_typed_observed_patterns_keep_their_authored_surface_maps():
    weathering = (ROOT / "king-street/app/weathering.ts").read_text()
    world_view = (ROOT / "king-street/app/world-view.ts").read_text()

    assert "token.includes('~obs1~') && /(?:^|~)pat=[^~]+/.test(token)" in weathering
    assert "if(observed)observedPatterns.add(m);else if(k)kinds.set(m,k)" in world_view
    assert "for(const m of observedPatterns)kinds.delete(m)" in world_view
    # The ordinary PBR map and normal-map stages remain in the material shader.
    assert "#include <normal_fragment_maps>" in weathering
    assert "diffuseColor.rgb=wxTint(wxSurface(diffuseColor.rgb))" in weathering
