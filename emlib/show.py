"""Embed rendered Manim scenes in notebooks, and render them.

Scenes live in animations/<module>/scenes.py and render to media/<module>/videos/scenes/<quality>/.
A notebook cell calls ``show.animation("m1_fields", "DivergenceVersusCurl")``. The default quality
is 1080p (Manim's -qh). If that video is missing, a lower-quality render is used if one exists;
otherwise the render command is printed. Set the environment variable ``EM_RENDER=1`` to render
missing scenes on the spot (slow for 3D scenes).

Render every scene from the repo root with ``uv run python -m emlib.show`` (add module names to
limit it, ``--quality l`` for quick 480p renders, ``--jobs N`` to render N modules at once).
"""

import ast
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
QUALITY = {"l": "480p15", "m": "720p30", "h": "1080p60"}
DEFAULT_QUALITY = "h"


def video_path(module, scene, quality=DEFAULT_QUALITY):
    return ROOT / "media" / module / "videos" / "scenes" / QUALITY[quality] / f"{scene}.mp4"


def render_command(module, scene, quality=DEFAULT_QUALITY):
    return f"uv run manim -q{quality} --media_dir media/{module} animations/{module}/scenes.py {scene}"


def render(module, scene, quality=DEFAULT_QUALITY):
    """Render a scene with the manim CLI. Returns the path of the video."""
    subprocess.run(render_command(module, scene, quality).split(), cwd=ROOT, check=True, capture_output=True)
    return video_path(module, scene, quality)


def animation(module, scene, quality=DEFAULT_QUALITY, width=720):
    """Return an embedded video for display in a notebook, rendering it first if asked to."""
    from IPython.display import HTML, Video

    path = video_path(module, scene, quality)
    if not path.exists() and os.environ.get("EM_RENDER") == "1":
        render(module, scene, quality)
    if not path.exists():
        lower = [q for q in ("h", "m", "l") if q != quality and video_path(module, scene, q).exists()]
        if lower:
            path = video_path(module, scene, lower[0])
    if path.exists():
        return Video(str(path), embed=True, width=width, html_attributes="controls loop")
    return HTML(
        f"<pre>video not rendered yet. From the repo root run:\n  {render_command(module, scene, quality)}\n"
        f"or set EM_RENDER=1 before running this cell.</pre>"
    )


def scenes(module):
    """Names of the Scene classes defined in animations/<module>/scenes.py, in file order."""
    tree = ast.parse((ROOT / "animations" / module / "scenes.py").read_text())
    classes = [n for n in tree.body if isinstance(n, ast.ClassDef)]
    is_scene = set()
    changed = True
    while changed:  # a class is a scene if a base is a Manim Scene or a scene class defined in this file
        changed = False
        for c in classes:
            bases = [ast.unparse(b) for b in c.bases]
            if c.name not in is_scene and any("Scene" in b or b in is_scene for b in bases):
                is_scene.add(c.name)
                changed = True
    return [c.name for c in classes if c.name in is_scene and not c.name.startswith("_")]


def modules():
    return sorted(p.parent.name for p in (ROOT / "animations").glob("*/scenes.py"))


def main(argv=None):
    import argparse

    parser = argparse.ArgumentParser(description="Render Manim scenes for the notebooks.")
    parser.add_argument("modules", nargs="*", help="module folders under animations/ (default: all)")
    parser.add_argument("--quality", default=DEFAULT_QUALITY, choices=sorted(QUALITY))
    parser.add_argument("--jobs", type=int, default=3, help="modules to render at once")
    parser.add_argument("--force", action="store_true", help="re-render scenes that already exist")
    args = parser.parse_args(argv)

    todo = [(m, s) for m in (args.modules or modules()) for s in scenes(m)]
    if not args.force:
        todo = [(m, s) for m, s in todo if not video_path(m, s, args.quality).exists()]
    print(f"{len(todo)} scene(s) to render at {QUALITY[args.quality]}", flush=True)

    # Scenes of one module share Manim's text and SVG cache under media/<module>, so two renders of the
    # same module at once can read a half-written cache file. Modules run in parallel, scenes in turn.
    by_module = {}
    for m, s in todo:
        by_module.setdefault(m, []).append(s)

    def module_job(m):
        lines = []
        for s in by_module[m]:
            try:
                render(m, s, args.quality)
                lines.append(f"ok      {m}/{s}")
            except subprocess.CalledProcessError as err:
                lines.append(f"FAILED  {m}/{s}: {err.stderr.decode(errors='replace').strip().splitlines()[-1:]}")
            print(lines[-1], flush=True)
        return lines

    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        results = list(pool.map(module_job, by_module))
    failed = sum(line.startswith("FAILED") for lines in results for line in lines)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
