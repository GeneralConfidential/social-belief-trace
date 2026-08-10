from __future__ import annotations

import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from npc_agent_benchmark.harness.runner import run_episode


@dataclass(frozen=True)
class Variant:
    variant_id: str
    use_provenance_weighting: bool
    enable_rumor_channel: bool
    apply_belief_updates: bool
    policy_mode: str


DEFAULT_TICKS_BY_SCENARIO: dict[str, int] = {
    "toy_scarcity_v0": 60,
    "toy_rumor_v0": 10,
    "toy_provenance_stress_v0": 70,
    "toy_contradiction_v0": 6,
    "toy_gossip_v0": 10,
    "toy_betrayal_v0": 8,
    "toy_help_stranger_v0": 24,
    "toy_dilemma_v0": 24,
    "toy_multihop_rumor_v0": 12,
    "toy_conflicting_sources_v0": 10,
    "toy_multihop_chain_v0": 18,
    "toy_adversarial_injection_v0": 12,
    "toy_escalating_rumors_v0": 22,
    "toy_social_pressure_v0": 32,
    # Non-game domain probes (BeliefBench general-first expansion)
    "pubhealth_conflicting_guidance_v0": 12,
    "workplace_policy_leak_v0": 12,
    "classroom_rumor_chain_v0": 18,
    "assistant_conflicting_advice_v0": 12,
    "disaster_route_misinfo_v0": 16,
    "newsroom_adversarial_tip_v0": 12,
}

VARIANTS: list[Variant] = [
    Variant("routine_weighted_update", True, True, True, "routine_only"),
    Variant("utility_no_belief_update", True, True, False, "utility_only"),
    Variant("utility_unweighted_belief_update", False, True, True, "utility_only"),
    Variant("utility_weighted_belief_update", True, True, True, "utility_only"),
    Variant("utility_no_communication", True, False, True, "utility_only"),
    Variant("utility_no_gossip_weighted_update", True, True, True, "utility_no_gossip"),
    Variant("gossip_driven_weighted_update", True, True, True, "gossip_driven"),
    Variant("reputation_probe_weighted_update", True, True, True, "reputation_probe"),
    Variant("external_stub_weighted_update", True, True, True, "external_stub"),
    Variant("reference_weighted_action_use", True, True, True, "reference"),
]

# Default seeds for aggregation (can be expanded later)
SEEDS = [0, 1, 2, 3, 4]


def _fmt(x: object) -> str:
    if x is None:
        return ""
    if isinstance(x, float):
        return f"{x:.3f}"
    return str(x)


def _mean(xs: list[float]) -> float | None:
    return (sum(xs) / len(xs)) if xs else None


def _std(xs: list[float]) -> float | None:
    if len(xs) < 2:
        return None
    m = sum(xs) / len(xs)
    v = sum((x - m) ** 2 for x in xs) / (len(xs) - 1)
    return math.sqrt(v)


def _get_nested(summary: dict[str, object], key: str) -> float | None:
    # supports one nested key used by the table: belief_uptake.trust_conf_pearson_r
    if key == "trust_conf_pearson_r":
        bu = summary.get("belief_uptake")
        if isinstance(bu, dict):
            v = bu.get("trust_conf_pearson_r")
            if isinstance(v, (int, float)):
                return float(v)
        return None
    if key == "trust_conf_pearson_r_n":
        bu = summary.get("belief_uptake")
        if isinstance(bu, dict):
            v = bu.get("trust_conf_pearson_r_n")
            if isinstance(v, (int, float)):
                return float(v)
        return None

    v = summary.get(key)
    if isinstance(v, (int, float)):
        return float(v)
    return None


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    scenarios_dir = root / "scenarios"
    runs_dir = root / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)

    raw_rows: list[dict[str, object]] = []

    for p in sorted(scenarios_dir.glob("*.json")):
        scenario = json.loads(p.read_text(encoding="utf-8"))
        sid = str(scenario.get("id", p.stem))
        ticks = int(DEFAULT_TICKS_BY_SCENARIO.get(sid, 24))

        for v in VARIANTS:
            for seed in SEEDS:
                out_path = runs_dir / f"{sid}__{v.variant_id}__seed{seed}.jsonl"
                summary = run_episode(
                    scenario=scenario,
                    ticks=ticks,
                    seed=seed,
                    trace_path=out_path,
                    use_provenance_weighting=v.use_provenance_weighting,
                    enable_rumor_channel=v.enable_rumor_channel,
                    apply_belief_updates=v.apply_belief_updates,
                    policy_mode=v.policy_mode,
                )
                summary["trace_path"] = out_path.relative_to(root).as_posix()
                summary["variant_id"] = v.variant_id
                summary["seed"] = seed
                raw_rows.append(summary)

    (runs_dir / "summary_raw.json").write_text(json.dumps(raw_rows, indent=2, sort_keys=True), encoding="utf-8")

    # Aggregate numeric metrics by (variant, scenario)
    metric_keys = [
        "avg_trust_drift_abs",
        "avg_trust_drift_signed",
        "belief_brier",
        "belief_truth_labeled_count",
        "belief_total_final",
        "belief_conf_delta_avg",
        "score_trace_arithmetic_validity",
        "help_stranger_count",
        "help_stranger_rate_per_event",
        "help_stranger_rate_per_peer_event",
        "value_break_count",
        "value_break_rate_per_event",
        "gossip_count",
        "gossip_rate_per_event",
        "belief_spread_fraction",
        "belief_spread_agents",
        "trust_conf_pearson_r",
        "trust_conf_pearson_r_n",
    ]

    buckets: dict[tuple[str, str], dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for r in raw_rows:
        vid = str(r.get("variant_id"))
        sid = str(r.get("scenario_id"))
        k = (vid, sid)
        for mk in metric_keys:
            val = _get_nested(r, mk)
            if val is not None:
                buckets[k][mk].append(val)

    agg_rows: list[dict[str, object]] = []
    for (vid, sid), vals in sorted(buckets.items()):
        row: dict[str, object] = {"variant_id": vid, "scenario_id": sid, "n_seeds": len(SEEDS)}
        for mk in metric_keys:
            row[f"{mk}_mean"] = _mean(vals.get(mk, []))
            row[f"{mk}_std"] = _std(vals.get(mk, []))
        agg_rows.append(row)

    (runs_dir / "summary_agg.json").write_text(json.dumps(agg_rows, indent=2, sort_keys=True), encoding="utf-8")

    # Markdown table
    cols = [
        "variant_id",
        "scenario_id",
        "n_seeds",
        "avg_trust_drift_abs_mean",
        "avg_trust_drift_abs_std",
        "avg_trust_drift_signed_mean",
        "belief_truth_labeled_count_mean",
        "belief_total_final_mean",
        "belief_brier_mean",
        "belief_brier_std",
        "trust_conf_pearson_r_mean",
        "trust_conf_pearson_r_n_mean",
        "help_stranger_count_mean",
        "help_stranger_rate_per_peer_event_mean",
        "value_break_count_mean",
        "value_break_rate_per_event_mean",
    ]

    lines = []
    lines.append("| " + " | ".join(cols) + " |")
    lines.append("| " + " | ".join(["---"] * len(cols)) + " |")
    for r in agg_rows:
        lines.append("| " + " | ".join(_fmt(r.get(c)) for c in cols) + " |")
    table = "\n".join(lines) + "\n"

    (runs_dir / "summary_agg.md").write_text(table, encoding="utf-8")
    print(table)


if __name__ == "__main__":
    main()

