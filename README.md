# Electromagnetics Study Patch

Jupyter notebooks and Manim animations for working through classical electromagnetism alongside
Malthe-Sørenssen, *Elementary Electromagnetism Using Python* (UiO FYS1120). The book is reference
only and is not part of this repo. No code or exercise solutions from the book appear here.

Work is grouped into six modules that bunch related book chapters. See [PLAN.md](PLAN.md) for the roadmap.

## Setup

Requires `uv`, `ffmpeg`, and the Cairo and Pango development libraries.

```
uv sync
```

## Run

```
uv run jupyter lab                      # notebooks
uv run pytest                           # emlib tests
uv run manim -ql --media_dir media/<module> animations/<module>/scenes.py <SceneName>   # render one scene at 480p
```

Rendered scenes are embedded in the matching notebook through `emlib.show.animation(module, scene)`.
If a scene has not been rendered the cell prints the render command. Set `EM_RENDER=1` to render on demand.
