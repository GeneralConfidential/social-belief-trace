# Provenance-stress diagnostic protocol (Cell A / Cell B)

**Goal.** Strengthen the paper’s central diagnostic: (A) auditable rerun of the interface confound with retained traces; (B) matched gossip permissions so symbolic and LLM policies share eligible targets and validation rules.

**Do not** silently alter historical `v0.3.0` tables. Report A/B as new experiments. Keep the disclosure that some pre-diagnostic LLM JSONL traces were not retained.

## Cell A — Interface confound (auditable)

| Item | Setting |
|---|---|
| Scenario | `toy_provenance_stress_v0` (unchanged) |
| Why empty peers | Single agent `a` ⇒ `observation.peer_ids == []` |
| Policies | `llm_openai` (or `llm_local`), `utility_only`, `reference` |
| Seeds | `{0,1,2,3,4}` (same as primary RQ tables) |
| Ticks | 70 |
| Retention | Keep JSONL under `runs/llm_eval/provenance_diagnostic/cell_a/` |
| Expected | LLM gossip structurally unavailable; symbolic may gossip to source/relationship ids |

## Cell B — Matched permissions

| Item | Setting |
|---|---|
| Scenario | `toy_provenance_stress_matched_gossip_v0` |
| Difference from A | Adds inert agent `peer`; `config.gossip_target_contract = peer_ids`; `peer` forced to `external_stub` |
| Same as A | Message schedule to `a`, trust sources, ticks, seeds, prompt version, uptake rules |
| Contract | Gossip/`help_stranger` targets must lie in `peer_ids` (same rule as LLM validator) |
| Focal metrics | Gossip/fallback counted for agent `a` only |
| Expected question | After equal permissions, does a communication gap remain? |

## Commands

```bash
# Pipeline (mock LLM, no API)
uv run python scripts/run_provenance_diagnostic.py --backend mock

# Paper cell (requires OPENAI_API_KEY)
uv run python scripts/run_provenance_diagnostic.py --backend openai --policy-modes llm_openai utility_only reference
```

## Outputs

- `runs/llm_eval/provenance_diagnostic/cell_{a,b}/*.jsonl`
- `summary_raw.json`, `summary_agg.json`, `summary.md`
- `autopsy.json` / `autopsy.md`
