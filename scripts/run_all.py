from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from npc_agent_benchmark.harness.runner import run_episode


@dataclass(frozen=True)
class RunConfig:
    ticks: int
    seed: int


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


def _fmt(x: object) -> str:
    if x is None:
        return ""
    if isinstance(x, float):
        return f"{x:.3f}"
    return str(x)


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    scenarios_dir = root / "scenarios"
    runs_dir = root / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)

    results: list[dict[str, object]] = []
    for p in sorted(scenarios_dir.glob("*.json")):
        scenario = json.loads(p.read_text(encoding="utf-8"))
        sid = str(scenario.get("id", p.stem))
        ticks = int(DEFAULT_TICKS_BY_SCENARIO.get(sid, 24))
        cfg = RunConfig(ticks=ticks, seed=0)

        for v in VARIANTS:
            out_path = runs_dir / f"{sid}__{v.variant_id}.jsonl"
            summary = run_episode(
                scenario=scenario,
                ticks=cfg.ticks,
                seed=cfg.seed,
                trace_path=out_path,
                use_provenance_weighting=v.use_provenance_weighting,
                enable_rumor_channel=v.enable_rumor_channel,
                apply_belief_updates=v.apply_belief_updates,
                policy_mode=v.policy_mode,
            )
            summary["trace_path"] = out_path.relative_to(root).as_posix()
            summary["variant_id"] = v.variant_id
            results.append(summary)

    # Write machine-readable summary
    (runs_dir / "summary.json").write_text(json.dumps(results, indent=2, sort_keys=True), encoding="utf-8")

    # Emit paper-friendly markdown table
    cols = [
        "variant_id",
        "scenario_id",
        "ticks",
        "avg_trust_drift_abs",
        "avg_trust_drift_signed",
        "belief_truth_labeled_count",
        "belief_total_final",
        "belief_brier",
        "trust_conf_pearson_r",
        "trust_conf_pearson_r_n",
        "belief_conf_delta_avg",
        "score_trace_arithmetic_validity",
        # source sensitivity lives under belief_uptake.trust_conf_pearson_r; we render it separately below
        "help_stranger_count",
        "help_stranger_rate_per_event",
        "help_stranger_rate_per_peer_event",
        "value_break_count",
        "value_break_rate_per_event",
        "gossip_count",
        "gossip_rate_per_event",
        "belief_spread_fraction",
        "belief_spread_agents",
    ]

    lines = []
    lines.append("| " + " | ".join(cols) + " |")
    lines.append("| " + " | ".join(["---"] * len(cols)) + " |")
    for r in results:
        uptake = r.get("belief_uptake", {})
        r_trust = None
        r_n = None
        if isinstance(uptake, dict):
            r_trust = uptake.get("trust_conf_pearson_r")
            r_n = uptake.get("trust_conf_pearson_r_n")
        r2 = dict(r)
        r2["trust_conf_pearson_r"] = r_trust
        r2["trust_conf_pearson_r_n"] = r_n
        lines.append("| " + " | ".join(_fmt(r2.get(c)) for c in cols) + " |")

    table = "\n".join(lines) + "\n"
    (runs_dir / "summary.md").write_text(table, encoding="utf-8")
    print(table)


if __name__ == "__main__":
    main()

