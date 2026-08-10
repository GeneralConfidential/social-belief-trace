# Launch Jupyter Lab for BeliefBench notebooks.
#
# Usage (from beliefbench/):
#   uv sync --group dev
#   uv run python scripts/run_notebooks.py
#   uv run python scripts/run_notebooks.py getting_started

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

BB_ROOT = Path(__file__).resolve().parents[1]
PUB_NB = BB_ROOT / "notebooks" / "getting_started.ipynb"
NB_DIR = BB_ROOT / "notebooks"


def main() -> None:
    parser = argparse.ArgumentParser(description="Open BeliefBench notebooks in Jupyter Lab.")
    parser.add_argument(
        "which",
        nargs="?",
        default="lab",
        choices=("lab", "getting_started"),
        help="lab=open notebooks folder; getting_started=open that notebook",
    )
    args = parser.parse_args()

    if not (BB_ROOT / ".venv").is_dir():
        print("No .venv found. Run first:\n  uv sync --group dev", file=sys.stderr)
        raise SystemExit(1)

    targets = [PUB_NB] if args.which == "getting_started" else [NB_DIR]

    for t in targets:
        if not t.exists():
            print(f"Missing: {t}", file=sys.stderr)
            raise SystemExit(1)

    cmd = [sys.executable, "-m", "jupyter", "lab", *[str(t) for t in targets]]
    print("Running:", " ".join(cmd))
    raise SystemExit(subprocess.call(cmd, cwd=str(BB_ROOT)))


if __name__ == "__main__":
    main()
