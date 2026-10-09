# SocialBeliefTrace

[![DOI](https://zenodo.org/badge/1329153233.svg)](https://doi.org/10.5281/zenodo.21877234)

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
| `notebooks/` | [`getting_started.ipynb`](notebooks/getting_started.ipynb) — walkthrough + basic harness checks |
| `ADAPTER_EXAMPLE.md` | How to plug in an external non-LLM policy |

## Quickstart

```bash
uv sync --group dev
uv run npc-agent-benchmark --scenario scenarios/toy_scarcity_v0.json --out runs/toy_scarcity_v0.jsonl --ticks 60 --seed 1
```

### Notebook

[`notebooks/getting_started.ipynb`](notebooks/getting_started.ipynb) imports the installed package, explains the harness/policy split, runs any scenario, and includes a few basic checks (trace schema, stub policy, registered modes).

```bash
uv sync --group dev
uv run python scripts/run_notebooks.py getting_started
# or: uv run jupyter lab notebooks/getting_started.ipynb
```

Use the project `.venv` as the Jupyter kernel. See [`notebooks/README.md`](notebooks/README.md) for setup tips.

Regenerate tables, figures, schema export, and validation:

```bash
uv run python scripts/regenerate.py
```

Smoke test and unit tests:

```bash
uv run python scripts/smoke_test.py
uv run pytest
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

**Software concept DOI** (always resolves to the latest Zenodo software version):  
[https://doi.org/10.5281/zenodo.21877234](https://doi.org/10.5281/zenodo.21877234)

**This software release (`v0.3.0`) version DOI:**  
[https://doi.org/10.5281/zenodo.23270279](https://doi.org/10.5281/zenodo.23270279)

**Companion preprint concept DOI** (always resolves to the latest Zenodo paper version):  
[https://doi.org/10.5281/zenodo.21937731](https://doi.org/10.5281/zenodo.21937731)

**This preprint cut version DOI:**  
[https://doi.org/10.5281/zenodo.23270303](https://doi.org/10.5281/zenodo.23270303)

Formerly released as **BeliefBench** `v0.1.0` ([https://doi.org/10.5281/zenodo.21877235](https://doi.org/10.5281/zenodo.21877235)). That name is independently used by Dixon's BeliefBench (LLM belief elicitation).

See also `CITATION.cff`.

## License

MIT — see `LICENSE`.
