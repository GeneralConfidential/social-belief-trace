"""Metric-validation cells: trusted-true vs false-flood on adversarial injection."""

from __future__ import annotations

import copy
import json
from pathlib import Path

from npc_agent_benchmark.harness.runner import run_episode

SCENARIO_ID = "toy_adversarial_injection_v0"
TICKS = 8
SEEDS = (0, 1, 2, 3, 4)
POLICY_MODES = ("utility_only", "reference")


def _trusted_true(base: dict) -> dict:
    sc = copy.deepcopy(base)
    recv = sc["agents"]["receiver"]
    recv["relationships"]["friend"]["trust"] = 0.9
    recv["relationships"]["hostile"]["trust"] = -0.9
    for ev in sc["events"]:
        if ev.get("from") == "friend":
            ev["confidence"] = 0.95
        elif ev.get("from") == "hostile":
            ev["confidence"] = 0.55
    sc["config"] = dict(sc.get("config") or {})
    sc["config"]["metric_validation_cell"] = "trusted_true"
    return sc


def _false_flood(base: dict) -> dict:
    """Repeated high-confidence false claims from a positively trusted hostile source."""
    sc = copy.deepcopy(base)
    recv = sc["agents"]["receiver"]
    recv["relationships"]["friend"]["trust"] = 0.2
    recv["relationships"]["hostile"]["trust"] = 0.6
    sc["events"] = [
        {"t": 1, "type": "message", "from": "hostile", "to": "receiver", "rumor_id": "route_blocked", "confidence": 0.95},
        {"t": 2, "type": "message", "from": "hostile", "to": "receiver", "rumor_id": "route_blocked", "confidence": 0.95},
        {"t": 3, "type": "message", "from": "hostile", "to": "receiver", "rumor_id": "route_blocked", "confidence": 0.95},
        {"t": 4, "type": "message", "from": "friend", "to": "receiver", "rumor_id": "route_clear", "confidence": 0.55},
    ]
    sc["config"] = dict(sc.get("config") or {})
    sc["config"]["metric_validation_cell"] = "false_flood"
    return sc


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    base = json.loads((root / "scenarios" / f"{SCENARIO_ID}.json").read_text(encoding="utf-8"))
    out_dir = root / "runs" / "metric_validation"
    out_dir.mkdir(parents=True, exist_ok=True)

    cells = {
        "trusted_true": _trusted_true(base),
        "false_flood": _false_flood(base),
    }
    rows: list[dict[str, object]] = []

    for cell_name, scenario in cells.items():
        for policy in POLICY_MODES:
            for seed in SEEDS:
                trace = out_dir / f"{SCENARIO_ID}__{cell_name}__{policy}__seed{seed}.jsonl"
                summary = run_episode(
                    scenario=scenario,
                    ticks=TICKS,
                    seed=seed,
                    trace_path=trace,
                    policy_mode=policy,
                )
                row = {
                    "cell": cell_name,
                    "policy_mode": policy,
                    "seed": seed,
                    "belief_accuracy": summary.get("belief_accuracy"),
                    "belief_brier": summary.get("belief_brier"),
                    "gossip_count": summary.get("gossip_count"),
                }
                uptake = summary.get("belief_uptake")
                if isinstance(uptake, dict):
                    row["trust_conf_pearson_r"] = uptake.get("trust_conf_pearson_r")
                    row["trust_conf_pearson_r_n"] = uptake.get("trust_conf_pearson_r_n")
                rows.append(row)
                print(
                    f"done {cell_name} {policy} seed={seed} "
                    f"acc={row.get('belief_accuracy')} brier={row.get('belief_brier')}",
                    flush=True,
                )

    (out_dir / "summary_raw.json").write_text(json.dumps(rows, indent=2, sort_keys=True), encoding="utf-8")

    # Aggregate
    from collections import defaultdict

    buckets: dict[tuple[str, str], dict[str, list[float]]] = defaultdict(
        lambda: {"belief_accuracy": [], "belief_brier": [], "gossip_count": []}
    )
    for r in rows:
        key = (str(r["cell"]), str(r["policy_mode"]))
        for k in ("belief_accuracy", "belief_brier", "gossip_count"):
            v = r.get(k)
            if isinstance(v, (int, float)):
                buckets[key][k].append(float(v))

    agg = []
    md = [
        "# Metric validation (trusted_true vs false_flood)",
        "",
        "| cell | policy | accuracy | brier | gossip |",
        "| --- | --- | --- | --- | --- |",
    ]
    for (cell, pol), vals in sorted(buckets.items()):
        row = {
            "cell": cell,
            "policy_mode": pol,
            "n_seeds": len(SEEDS),
        }
        for k, xs in vals.items():
            row[f"{k}_mean"] = (sum(xs) / len(xs)) if xs else None
        agg.append(row)
        md.append(
            f"| {cell} | {pol} | {row.get('belief_accuracy_mean')} | "
            f"{row.get('belief_brier_mean')} | {row.get('gossip_count_mean')} |"
        )

    (out_dir / "summary_agg.json").write_text(json.dumps(agg, indent=2, sort_keys=True), encoding="utf-8")
    (out_dir / "summary_agg.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print((out_dir / "summary_agg.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
