"""Embed rendered Manim scenes in notebooks.

Scenes live in animations/<module>/scenes.py and render to media/<module>/videos/scenes/<quality>/.
A notebook cell calls ``show.animation("m1_fields", "DivergenceVersusCurl")``. If the video exists
it is embedded; otherwise the render command is printed. Set the environment variable
``EM_RENDER=1`` to render missing scenes on the spot (slow for 3D scenes).
"""

import os
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
QUALITY = {"l": "480p15", "m": "720p30", "h": "1080p60"}


def video_path(module, scene, quality="l"):
    return ROOT / "media" / module / "videos" / "scenes" / QUALITY[quality] / f"{scene}.mp4"


def render_command(module, scene, quality="l"):
    return f"uv run manim -q{quality} --media_dir media/{module} animations/{module}/scenes.py {scene}"


def render(module, scene, quality="l"):
    """Render a scene with the manim CLI. Returns the path of the video."""
    subprocess.run(render_command(module, scene, quality).split(), cwd=ROOT, check=True, capture_output=True)
    return video_path(module, scene, quality)


def animation(module, scene, quality="l", width=720):
    """Return an embedded video for display in a notebook, rendering it first if asked to."""
    from IPython.display import HTML, Video

    path = video_path(module, scene, quality)
    if not path.exists() and os.environ.get("EM_RENDER") == "1":
        render(module, scene, quality)
    if path.exists():
        return Video(str(path), embed=True, width=width, html_attributes="controls loop")
    return HTML(
        f"<pre>video not rendered yet. From the repo root run:\n  {render_command(module, scene, quality)}\n"
        f"or set EM_RENDER=1 before running this cell.</pre>"
    )
