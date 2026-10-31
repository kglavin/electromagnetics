from emlib import show


def test_scene_discovery_follows_inheritance_and_skips_helpers():
    m1 = show.scenes("m1_fields")
    assert "InternalFacesCancel" in m1 and "DotProductShadow" in m1  # subclasses of private bases
    assert not any(name.startswith("_") for name in m1)
    assert sum(len(show.scenes(m)) for m in show.modules()) == 28


def test_default_quality_is_1080p():
    assert show.DEFAULT_QUALITY == "h"
    assert "-qh" in show.render_command("m1_fields", "DivergenceVersusCurl")
    assert show.video_path("m1_fields", "DivergenceVersusCurl").parent.name == "1080p60"
