"""Thin misinformation stress matrix on adversarial injection (symbolic + optional LLM)."""

from __future__ import annotations

import copy
import json
from itertools import product
from pathlib import Path

from npc_agent_benchmark.harness.runner import run_episode

SCENARIO_ID = "toy_adversarial_injection_v0"
TICKS = 8
SEEDS = (0, 1, 2)
# trust(hostile), trust(friend), conf(false/hostile msg), conf(true/friend msg)
TRUST_HOSTILE = (-0.8, 0.0, 0.5)
TRUST_FRIEND = (0.75, 0.0)
CONF_FALSE = (0.55, 0.95)
CONF_TRUE = (0.55, 0.75)
POLICY_MODES = ("utility_only", "reference")


def _mutate(
    base: dict,
    *,
    trust_hostile: float,
    trust_friend: float,
    conf_false: float,
    conf_true: float,
) -> dict:
    sc = copy.deepcopy(base)
    recv = sc["agents"]["receiver"]
    recv["relationships"]["hostile"]["trust"] = trust_hostile
    recv["relationships"]["friend"]["trust"] = trust_friend
    for ev in sc["events"]:
        if ev.get("from") == "hostile":
            ev["confidence"] = conf_false
        elif ev.get("from") == "friend":
            ev["confidence"] = conf_true
    sc["config"] = dict(sc.get("config") or {})
    sc["config"]["stress_cell"] = {
        "trust_hostile": trust_hostile,
        "trust_friend": trust_friend,
        "conf_false": conf_false,
        "conf_true": conf_true,
    }
    return sc


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    base = json.loads((root / "scenarios" / f"{SCENARIO_ID}.json").read_text(encoding="utf-8"))
    out_dir = root / "runs" / "stress_matrix"
    out_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []

    cells = list(product(TRUST_HOSTILE, TRUST_FRIEND, CONF_FALSE, CONF_TRUE))
    print(f"cells={len(cells)} policies={len(POLICY_MODES)} seeds={len(SEEDS)}")

    for th, tf, cf, ct in cells:
        scenario = _mutate(base, trust_hostile=th, trust_friend=tf, conf_false=cf, conf_true=ct)
        cell_id = f"th{th}_tf{tf}_cf{cf}_ct{ct}".replace(".", "p").replace("-", "m")
        for policy in POLICY_MODES:
            for seed in SEEDS:
                trace = out_dir / f"{SCENARIO_ID}__{cell_id}__{policy}__seed{seed}.jsonl"
                summary = run_episode(
                    scenario=scenario,
                    ticks=TICKS,
                    seed=seed,
                    trace_path=trace,
                    policy_mode=policy,
                    use_provenance_weighting=True,
                    enable_rumor_channel=True,
                    apply_belief_updates=True,
                )
                uptake = summary.get("belief_uptake") if isinstance(summary.get("belief_uptake"), dict) else {}
                rows.append(
                    {
                        "cell_id": cell_id,
                        "trust_hostile": th,
                        "trust_friend": tf,
                        "conf_false": cf,
                        "conf_true": ct,
                        "policy_mode": policy,
                        "seed": seed,
                        "belief_accuracy": summary.get("belief_accuracy"),
                        "belief_brier": summary.get("belief_brier"),
                        "belief_total_final": summary.get("belief_total_final"),
                        "trust_conf_pearson_r": uptake.get("trust_conf_pearson_r"),
                        "gossip_count": summary.get("gossip_count"),
                        "avg_trust_drift_signed": summary.get("avg_trust_drift_signed"),
                    }
                )

    raw_path = out_dir / "summary_raw.json"
    raw_path.write_text(json.dumps(rows, indent=2, sort_keys=True), encoding="utf-8")

    # Aggregate by cell + policy
    buckets: dict[tuple[str, str], list[dict[str, object]]] = {}
    for row in rows:
        key = (str(row["cell_id"]), str(row["policy_mode"]))
        buckets.setdefault(key, []).append(row)

    agg = []
    for (cell_id, policy), group in sorted(buckets.items()):
        def mean(k: str) -> float | None:
            xs = [float(g[k]) for g in group if isinstance(g.get(k), (int, float))]
            return (sum(xs) / len(xs)) if xs else None

        head = group[0]
        agg.append(
            {
                "cell_id": cell_id,
                "policy_mode": policy,
                "trust_hostile": head["trust_hostile"],
                "trust_friend": head["trust_friend"],
                "conf_false": head["conf_false"],
                "conf_true": head["conf_true"],
                "n_seeds": len(group),
                "belief_accuracy_mean": mean("belief_accuracy"),
                "belief_brier_mean": mean("belief_brier"),
                "trust_conf_pearson_r_mean": mean("trust_conf_pearson_r"),
                "gossip_count_mean": mean("gossip_count"),
            }
        )

    (out_dir / "summary_agg.json").write_text(json.dumps(agg, indent=2, sort_keys=True), encoding="utf-8")

    md = [
        "# Misinformation stress matrix (toy_adversarial_injection_v0)",
        "",
        "| policy | th | tf | cf | ct | accuracy | brier | gossip |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for r in agg:
        md.append(
            f"| {r['policy_mode']} | {r['trust_hostile']} | {r['trust_friend']} | "
            f"{r['conf_false']} | {r['conf_true']} | {r.get('belief_accuracy_mean')} | "
            f"{r.get('belief_brier_mean')} | {r.get('gossip_count_mean')} |"
        )
    md_path = out_dir / "summary_agg.md"
    md_path.write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"Wrote {len(rows)} rows, {len(agg)} aggregates -> {md_path}")


if __name__ == "__main__":
    main()
