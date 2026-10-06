"""Symbolic belief-decay multiplier ablation (no LLM / no API)."""

from __future__ import annotations

import copy
import json
from pathlib import Path

from npc_agent_benchmark.harness.runner import run_episode

SCENARIO_ID = "toy_rumor_v0"
TICKS = 10
SEEDS = (0, 1, 2, 3, 4)
MULTIPLIERS = (0.0, 1.0, 4.0)
POLICY = "utility_only"
METRICS = (
    "belief_total_final",
    "belief_count",
    "belief_conf_delta_avg",
    "belief_brier",
    "belief_accuracy",
    "gossip_count",
)


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    base = json.loads((root / "scenarios" / f"{SCENARIO_ID}.json").read_text(encoding="utf-8"))
    out_dir = root / "runs" / "decay_ablation"
    out_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []

    for mult in MULTIPLIERS:
        for seed in SEEDS:
            scenario = copy.deepcopy(base)
            cfg = scenario.setdefault("config", {})
            if not isinstance(cfg, dict):
                cfg = {}
                scenario["config"] = cfg
            cfg["belief_decay_multiplier"] = mult
            trace = out_dir / f"{SCENARIO_ID}__decay{mult}__seed{seed}.jsonl"
            summary = run_episode(
                scenario=scenario,
                ticks=TICKS,
                seed=seed,
                trace_path=trace,
                policy_mode=POLICY,
                use_provenance_weighting=True,
                enable_rumor_channel=True,
                apply_belief_updates=True,
            )
            row: dict[str, object] = {
                "scenario_id": SCENARIO_ID,
                "policy_mode": POLICY,
                "belief_decay_multiplier": mult,
                "seed": seed,
            }
            for mk in METRICS:
                row[mk] = summary.get(mk)
            rows.append(row)

    raw_path = out_dir / "summary_raw.json"
    raw_path.write_text(json.dumps(rows, indent=2, sort_keys=True), encoding="utf-8")

    buckets: dict[float, dict[str, list[float]]] = {}
    for row in rows:
        m = float(row["belief_decay_multiplier"])  # type: ignore[arg-type]
        bucket = buckets.setdefault(m, {mk: [] for mk in METRICS})
        for mk in METRICS:
            val = row.get(mk)
            if isinstance(val, (int, float)):
                bucket[mk].append(float(val))

    agg = []
    for m in sorted(buckets):
        agg_row: dict[str, object] = {
            "belief_decay_multiplier": m,
            "n_seeds": len(SEEDS),
            "scenario_id": SCENARIO_ID,
            "policy_mode": POLICY,
        }
        for mk in METRICS:
            xs = buckets[m][mk]
            agg_row[f"{mk}_mean"] = (sum(xs) / len(xs)) if xs else None
        agg.append(agg_row)

    agg_path = out_dir / "summary_agg.json"
    agg_path.write_text(json.dumps(agg, indent=2, sort_keys=True), encoding="utf-8")
    md = [
        "| decay_mult | belief_total_final_mean | belief_conf_delta_avg_mean | gossip_count_mean |",
        "| --- | --- | --- | --- |",
    ]
    for r in agg:
        md.append(
            f"| {r['belief_decay_multiplier']} | {r.get('belief_total_final_mean')} | "
            f"{r.get('belief_conf_delta_avg_mean')} | {r.get('gossip_count_mean')} |"
        )
    md_path = out_dir / "summary_agg.md"
    md_path.write_text("\n".join(md) + "\n", encoding="utf-8")
    print(md_path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
