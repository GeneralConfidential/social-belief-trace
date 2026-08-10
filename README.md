# BeliefBench

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21877235.svg)](https://doi.org/10.5281/zenodo.21877235)

Evaluation harness for misinformation-aware social agents: inspectable needs, trust, and source-confidence beliefs.

The package ships schema-versioned JSONL traces (`trace_v0`), diagnostic scenarios, and metrics for trust drift, calibration, source sensitivity, and related checks. Bundled agents are small symbolic baselines used to validate the harness and ablations—not a claim of social-intelligence SOTA.

## Layout

| Path | Role |
|------|------|
| `scenarios/` | Scenario JSON specs |
| `src/` | Harness, metrics, baselines |
| `scripts/` | Run, aggregate, plot, regenerate, validate |
| `schemas/` | JSON Schema for `trace_v0` |
| `runs/` | Summaries / figures (regenerate locally) |
| `tests/` | Pytest suite |
| `notebooks/` | Getting-started notebook |
| `ADAPTER_EXAMPLE.md` | How to plug in an external non-LLM policy |

## Quickstart

```bash
uv sync --group dev
uv run npc-agent-benchmark --scenario scenarios/toy_scarcity_v0.json --out runs/toy_scarcity_v0.jsonl --ticks 60 --seed 1
```

Regenerate tables, figures, schema export, and validation:

```bash
uv run python scripts/regenerate.py
```

Smoke test and unit tests:

```bash
uv run python scripts/smoke_test.py
uv run pytest
```

Optional notebook:

```bash
uv sync --group dev
uv run python scripts/run_notebooks.py getting_started
```

Expected full regeneration is under about a minute on a typical laptop CPU.

## Reproduction checklist

```bash
uv sync --group dev
uv run python scripts/smoke_test.py
uv run python scripts/regenerate.py
uv run python scripts/validate_outputs.py
uv run pytest
```

Environment: Python 3.12+ via `uv` (`pyproject.toml` / `uv.lock`). Paths in summary JSON are relative to this folder.

## Citation

Archived release: [https://doi.org/10.5281/zenodo.21877235](https://doi.org/10.5281/zenodo.21877235) (`v0.1.0`).  
See also `CITATION.cff`.

## License

MIT — see `LICENSE`.
