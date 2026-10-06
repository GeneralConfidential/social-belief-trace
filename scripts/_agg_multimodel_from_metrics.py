"""Rebuild multimodel summary_agg from existing *.metrics.json sidecars."""

from __future__ import annotations

import json
from pathlib import Path

METRIC_KEYS = (
    "gossip_count",
    "gossip_rate_per_event",
    "belief_spread_fraction",
    "help_stranger_rate_per_peer_event",
    "llm_invalid_action_rate",
    "belief_accuracy",
    "belief_brier",
    "avg_trust_drift_abs",
)
UPTAKE_KEYS = ("trust_conf_pearson_r", "trust_conf_pearson_r_n")
SEEDS = (0, 1, 2)


def main() -> None:
    out = Path(__file__).resolve().parents[1] / "runs" / "llm_eval" / "multimodel"
    rows = [json.loads(p.read_text(encoding="utf-8")) for p in sorted(out.glob("*.metrics.json"))]
    print(f"rows={len(rows)}")
    (out / "summary_raw.json").write_text(json.dumps(rows, indent=2, sort_keys=True), encoding="utf-8")

    buckets: dict[tuple[str, str], dict[str, list[float]]] = {}
    keys = list(METRIC_KEYS) + list(UPTAKE_KEYS)
    for row in rows:
        key = (str(row["model_label"]), str(row["scenario_id"]))
        bucket = buckets.setdefault(key, {k: [] for k in keys})
        for k in keys:
            val = row.get(k)
            if isinstance(val, (int, float)):
                bucket[k].append(float(val))

    agg: list[dict[str, object]] = []
    for (label, sid), vals in sorted(buckets.items()):
        agg_row: dict[str, object] = {
            "model_label": label,
            "scenario_id": sid,
            "n_seeds": len(SEEDS),
        }
        for k in keys:
            xs = vals[k]
            agg_row[f"{k}_mean"] = (sum(xs) / len(xs)) if xs else None
        agg.append(agg_row)

    (out / "summary_agg.json").write_text(json.dumps(agg, indent=2, sort_keys=True), encoding="utf-8")
    md = [
        "# Multi-model RQ6 contrast",
        "",
        "| model | scenario | gossip | invalid | trust_conf_r | belief_spread | accuracy |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for r in agg:
        md.append(
            f"| {r['model_label']} | {r['scenario_id']} | "
            f"{r.get('gossip_count_mean')} | {r.get('llm_invalid_action_rate_mean')} | "
            f"{r.get('trust_conf_pearson_r_mean')} | {r.get('belief_spread_fraction_mean')} | "
            f"{r.get('belief_accuracy_mean')} |"
        )
    path = out / "summary_agg.md"
    path.write_text("\n".join(md) + "\n", encoding="utf-8")
    print(path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
