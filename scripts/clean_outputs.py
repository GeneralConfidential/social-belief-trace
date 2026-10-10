from __future__ import annotations

import shutil
from pathlib import Path


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    runs_dir = root / "runs"
    # Preserve committed LLM / ablation summaries (not rebuilt by regenerate.py).
    preserve = {
        "llm_eval",
        "decay_ablation",
        "metric_validation",
        "stress_matrix",
    }
    if runs_dir.exists():
        for child in list(runs_dir.iterdir()):
            if child.name in preserve:
                continue
            if child.is_dir():
                shutil.rmtree(child)
            else:
                child.unlink()
    runs_dir.mkdir(parents=True, exist_ok=True)
    print("Cleaned symbolic runs/ outputs (preserved llm_eval and ablation summaries).")


if __name__ == "__main__":
    main()
