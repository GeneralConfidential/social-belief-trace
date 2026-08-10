from __future__ import annotations

import json
from pathlib import Path

from npc_agent_benchmark.models import TraceEvent


EXPECTED_FILES = [
    "runs/summary.json",
    "runs/summary.md",
    "runs/summary_raw.json",
    "runs/summary_agg.json",
    "runs/summary_agg.md",
    "runs/figures/fig1_brier_provenance_stress.png",
    "runs/figures/fig2_source_sensitivity_provenance_stress.png",
    "schemas/trace_v0.schema.json",
]

EXPECTED_FIGURES = {
    "fig1_brier_provenance_stress.png",
    "fig2_source_sensitivity_provenance_stress.png",
}

TRACE_SCHEMA_VERSION = "trace_v0"


def _load_json(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def _assert_relative_trace_paths(rows: list[dict[str, object]], *, source: str) -> None:
    for i, row in enumerate(rows):
        trace_path = row.get("trace_path")
        if not isinstance(trace_path, str):
            raise ValueError(f"{source}[{i}] missing string trace_path")
        path = Path(trace_path)
        if path.is_absolute() or ":" in trace_path:
            raise ValueError(f"{source}[{i}] has non-portable trace_path: {trace_path}")
        # Portable summaries should use POSIX separators (forward slashes), not Windows backslashes.
        if "\\" in trace_path:
            raise ValueError(f"{source}[{i}] trace_path must use forward slashes: {trace_path}")


def _assert_summary_agg_rows(path: Path) -> None:
    """Structural check for aggregated summaries (no trace_path on these rows)."""
    data = _load_json(path)
    if not isinstance(data, list) or not data:
        raise ValueError(f"{path.name} must be a non-empty JSON list")
    required = ("variant_id", "scenario_id", "n_seeds")
    for i, row in enumerate(data):
        if not isinstance(row, dict):
            raise ValueError(f"{path.name}[{i}] must be an object")
        for k in required:
            if k not in row:
                raise ValueError(f"{path.name}[{i}] missing {k!r}")


def _assert_trace_schema_every_line(root: Path, trace_path: str) -> None:
    """Every non-empty JSONL line must carry trace_schema_version (not only the first)."""
    path = root / trace_path
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            stripped = line.strip()
            if not stripped:
                continue
            event = TraceEvent.model_validate_json(stripped)
            if event.trace_schema_version != TRACE_SCHEMA_VERSION:
                raise ValueError(
                    f"{trace_path}:{line_no} expected {TRACE_SCHEMA_VERSION}, "
                    f"got {event.trace_schema_version!r}"
                )


def _assert_all_traces_schema_consistent(root: Path, rows: list[dict[str, object]], *, source: str) -> None:
    paths: set[str] = set()
    for row in rows:
        tp = row.get("trace_path")
        if isinstance(tp, str):
            paths.add(tp)
    for rel in sorted(paths):
        _assert_trace_schema_every_line(root, rel)


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    missing = [p for p in EXPECTED_FILES if not (root / p).exists()]
    if missing:
        raise SystemExit("Missing expected files:\n" + "\n".join(f"- {p}" for p in missing))

    figures_dir = root / "runs" / "figures"
    actual_figures = {p.name for p in figures_dir.glob("*.png")}
    extra_figures = sorted(actual_figures - EXPECTED_FIGURES)
    if extra_figures:
        raise SystemExit("Unexpected stale figures:\n" + "\n".join(f"- {p}" for p in extra_figures))

    summary = _load_json(root / "runs" / "summary.json")
    summary_raw = _load_json(root / "runs" / "summary_raw.json")

    if not isinstance(summary, list) or not summary:
        raise SystemExit("runs/summary.json must be a non-empty JSON list")
    if not isinstance(summary_raw, list) or not summary_raw:
        raise SystemExit("runs/summary_raw.json must be a non-empty JSON list")

    _assert_relative_trace_paths(summary, source="summary")
    _assert_relative_trace_paths(summary_raw, source="summary_raw")
    _assert_all_traces_schema_consistent(root, summary, source="summary")
    _assert_all_traces_schema_consistent(root, summary_raw, source="summary_raw")
    _assert_summary_agg_rows(root / "runs" / "summary_agg.json")

    print("Validated generated benchmark artifacts.")


if __name__ == "__main__":
    main()
