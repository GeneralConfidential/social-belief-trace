from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def _run(script: str) -> None:
    root = Path(__file__).resolve().parents[1]
    subprocess.run([sys.executable, str(root / "scripts" / script)], cwd=root, check=True)


def main() -> None:
    _run("clean_outputs.py")
    _run("run_all_agg.py")
    _run("plot_results.py")
    _run("run_all.py")
    _run("export_trace_schema.py")
    _run("validate_outputs.py")


if __name__ == "__main__":
    main()
