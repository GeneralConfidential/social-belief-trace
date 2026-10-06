"""Prompt sensitivity: llm_act_v2 vs llm_act_v2_needfirst (OpenAI, rumor + provenance)."""

from __future__ import annotations

import json
import os
from pathlib import Path

from npc_agent_benchmark.baselines.llm_client import OpenAIClient
from npc_agent_benchmark.baselines.llm_prompt import PROMPT_VERSION, PROMPT_VERSION_NEEDFIRST
from npc_agent_benchmark.harness.runner import run_episode

SCENARIOS = ("toy_rumor_v0", "toy_provenance_stress_v0")
SEEDS = (0, 1, 2)
TICKS = {"toy_rumor_v0": 10, "toy_provenance_stress_v0": 70}
PROMPTS = (PROMPT_VERSION, PROMPT_VERSION_NEEDFIRST)
METRICS = (
    "gossip_count",
    "gossip_rate_per_event",
    "llm_invalid_action_rate",
    "llm_act_successes",
    "llm_act_fallbacks",
)


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    out_dir = root / "runs" / "llm_eval" / "prompt_sensitivity"
    out_dir.mkdir(parents=True, exist_ok=True)
    client = OpenAIClient.from_env()
    rows: list[dict[str, object]] = []

    for prompt_version in PROMPTS:
        os.environ["SBT_LLM_PROMPT_VERSION"] = prompt_version
        for sid in SCENARIOS:
            scenario = json.loads((root / "scenarios" / f"{sid}.json").read_text(encoding="utf-8"))
            ticks = TICKS[sid]
            for seed in SEEDS:
                trace = out_dir / f"{sid}__{prompt_version}__seed{seed}.jsonl"
                summary = run_episode(
                    scenario=scenario,
                    ticks=ticks,
                    seed=seed,
                    trace_path=trace,
                    policy_mode="llm_openai",
                    llm_client=client,
                )
                row = {
                    "scenario_id": sid,
                    "prompt_version": prompt_version,
                    "seed": seed,
                    "backend": "openai",
                    "trace_path": trace.relative_to(root).as_posix(),
                }
                for mk in METRICS:
                    row[mk] = summary.get(mk)
                row["action_kind_counts"] = summary.get("action_kind_counts")
                rows.append(row)
                print(f"done {sid} {prompt_version} seed={seed} gossip={row.get('gossip_count')} invalid={row.get('llm_invalid_action_rate')}")

    raw_path = out_dir / "summary_raw.json"
    raw_path.write_text(json.dumps(rows, indent=2, sort_keys=True), encoding="utf-8")

    # Aggregate means by (scenario, prompt)
    buckets: dict[tuple[str, str], dict[str, list[float]]] = {}
    for row in rows:
        key = (str(row["scenario_id"]), str(row["prompt_version"]))
        bucket = buckets.setdefault(key, {mk: [] for mk in METRICS})
        for mk in METRICS:
            val = row.get(mk)
            if isinstance(val, (int, float)):
                bucket[mk].append(float(val))

    agg = []
    for (sid, pv), vals in sorted(buckets.items()):
        agg_row: dict[str, object] = {"scenario_id": sid, "prompt_version": pv, "n_seeds": len(SEEDS)}
        for mk in METRICS:
            xs = vals[mk]
            agg_row[f"{mk}_mean"] = (sum(xs) / len(xs)) if xs else None
        agg.append(agg_row)

    agg_path = out_dir / "summary_agg.json"
    agg_path.write_text(json.dumps(agg, indent=2, sort_keys=True), encoding="utf-8")
    md = ["| scenario | prompt | gossip_mean | invalid_mean |", "| --- | --- | --- | --- |"]
    for r in agg:
        md.append(
            f"| {r['scenario_id']} | {r['prompt_version']} | "
            f"{r.get('gossip_count_mean')} | {r.get('llm_invalid_action_rate_mean')} |"
        )
    md_path = out_dir / "summary_agg.md"
    md_path.write_text("\n".join(md) + "\n", encoding="utf-8")
    print(md_path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
