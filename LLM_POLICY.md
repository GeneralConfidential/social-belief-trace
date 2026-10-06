# LLM policy (opt-in)

Constrained language-model policies implement the same `observe/act` contract as symbolic baselines. Belief uptake, trust, and contradiction handling remain in the harness.

## Policy modes

| Mode | Backend | When to use |
|------|---------|-------------|
| `llm_local` | [Ollama](https://ollama.com) HTTP API | Local dev / smoke on your machine |
| `llm_openai` | OpenAI chat completions | Paper RQ6 tables (pinned model) |

**Not included in** `scripts/regenerate.py` or default CI (symbolic + mock only).

## Environment variables

### Ollama (`llm_local`)

| Variable | Default |
|----------|---------|
| `SBT_OLLAMA_BASE_URL` | `http://localhost:11434` |
| `SBT_OLLAMA_MODEL` | `llama3.2:3b-instruct-q4_K_M` |
| `SBT_OLLAMA_TEMPERATURE` | `0` |
| `SBT_OLLAMA_TIMEOUT_S` | `120` |

### OpenAI (`llm_openai`)

| Variable | Default |
|----------|---------|
| `OPENAI_API_KEY` | *(required)* |
| `SBT_OPENAI_MODEL` | `gpt-4o-mini` |
| `SBT_OPENAI_TEMPERATURE` | `0` (set `omit` for models that reject `temperature=0`, e.g. `gpt-6.1-sol`) |
| `SBT_OPENAI_TIMEOUT_S` | `120` |
| `SBT_OPENAI_BASE_URL` | `https://api.openai.com/v1` |

### Prompt pinning

- **Prompt version:** `llm_act_v2` default (see `baselines/llm_prompt.py` and manuscript appendix). Override with `SBT_LLM_PROMPT_VERSION` (e.g. `llm_act_v2_needfirst` for need-first sensitivity).
- **Ollama:** requests use `format=json`; invalid outputs get one repair pass then `rest` fallback.
- **Invalid actions:** parse/validation failures → repair attempt → `rest`; rates in episode summary (`llm_invalid_action_rate`, `llm_act_repairs`, etc.).
- **Analyses:** `scripts/analyze_rq6_autopsy.py`, `scripts/run_prompt_sensitivity.py`, `scripts/run_decay_ablation.py`.

## Commands

```bash
# Mock pipeline (no network)
uv run python scripts/run_llm_eval.py --mock

# Local Ollama (after: ollama pull llama3.2:3b-instruct-q4_K_M)
uv run python scripts/llm_smoke.py

# OpenAI RQ6 slice (~$1–2 on gpt-4o-mini for default scenarios/seeds)
export OPENAI_API_KEY=sk-...
uv run python scripts/run_llm_eval.py --backend openai --policy-modes llm_openai utility_only reference
# Writes runs/llm_eval/summary_agg_openai.json (Ollama summaries stay in summary_agg.json)

# Multi-model / H-tier / Sol slices (resume via *.metrics.json)
uv run python scripts/run_multimodel_rq6.py
uv run python scripts/run_multimodel_h.py
SBT_OPENAI_TEMPERATURE=omit SBT_OPENAI_MODEL=gpt-6.1-sol uv run python scripts/run_multimodel_sol.py

# Single episode via CLI
uv run npc-agent-benchmark --scenario scenarios/toy_rumor_v0.json \
  --out runs/llm_rumor.jsonl --ticks 10 --seed 0 --policy-mode llm_local
```

Outputs: `runs/llm_eval/summary_raw.json`, `summary_agg.json`, `summary_agg.md`.

## Tests

```bash
uv run python -m pytest -q
```

LLM tests use `MockLlmClient` only.
