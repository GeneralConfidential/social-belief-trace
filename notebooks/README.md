# Notebooks

| Notebook | Notes |
|----------|-------|
| [`getting_started.ipynb`](getting_started.ipynb) | Imports the package, runs a scenario, includes a few basic harness checks |

## Setup

```bash
cd beliefbench
uv sync --group dev
uv run python -m ipykernel install --user --name=beliefbench --display-name="BeliefBench (uv)"
```

## Run

```bash
uv run python scripts/run_notebooks.py getting_started
# or:
uv run jupyter lab notebooks/getting_started.ipynb
```

Select the project `.venv` (or the **BeliefBench (uv)** kernel) when prompted.
