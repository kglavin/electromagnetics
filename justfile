# Tasks for the electromagnetics study patch. Run `just` to list them.
# Install just with: uv tool install rust-just

set shell := ["bash", "-euo", "pipefail", "-c"]

# list the recipes
default:
    @just --list

# install or update the environment
sync:
    uv sync

# run the emlib tests
test *args:
    uv run pytest -q {{args}}

# open JupyterLab
lab:
    uv run jupyter lab

# render scenes at 1080p that are not rendered yet (all modules, or the ones named)
render *modules:
    uv run python -m emlib.show {{modules}}

# render quick 480p versions
render-quick *modules:
    uv run python -m emlib.show --quality l {{modules}}

# re-render scenes at 1080p even if they exist
render-force *modules:
    uv run python -m emlib.show --force {{modules}}

# render one scene, e.g. just scene m4_magnetostatics HelmholtzFieldLines
scene module name quality="h":
    uv run manim -q{{quality}} --media_dir media/{{module}} animations/{{module}}/scenes.py {{name}}

# execute notebooks top to bottom into a scratch folder (all, or a folder or file); repo files keep no outputs
check-notebooks path="notebooks":
    #!/usr/bin/env bash
    set -euo pipefail
    out=$(mktemp -d)
    fail=0
    for nb in $(find {{path}} -name '*.ipynb' -not -path '*/.ipynb_checkpoints/*' | sort); do
        start=$(date +%s)
        if uv run jupyter nbconvert --to notebook --execute "$nb" --output-dir "$out" --ExecutePreprocessor.timeout=900 >/dev/null 2>"$out/err.txt"; then
            echo "ok    $(( $(date +%s) - start ))s  $nb"
        else
            echo "FAIL  $nb"; tail -5 "$out/err.txt"; fail=1
        fi
    done
    echo "executed copies in $out"
    exit $fail

# tests plus every notebook
check: test check-notebooks
