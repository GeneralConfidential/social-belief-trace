# Notebooks

| Notebook | Notes |
|----------|-------|
| [`getting_started.ipynb`](getting_started.ipynb) | Imports the package, runs a scenario, includes a few basic harness checks |

## Setup

```bash
cd social-belief-trace
uv sync --group dev
uv run python -m ipykernel install --user --name=social-belief-trace --display-name="SocialBeliefTrace (uv)"
```

## Run

```bash
uv run python scripts/run_notebooks.py getting_started
# or:
uv run jupyter lab notebooks/getting_started.ipynb
```

Select the project `.venv` (or the **SocialBeliefTrace (uv)** kernel) when prompted.
