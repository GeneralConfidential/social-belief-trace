"""Optional H-tier multimodel slice: gpt-4o + stronger open model (prep / opt-in)."""

from __future__ import annotations

import json
import os
from pathlib import Path

from npc_agent_benchmark.baselines.llm_client import OllamaClient, OpenAIClient
from npc_agent_benchmark.harness.runner import run_episode

# Keep small: same scenarios as multimodel contrast, 3 seeds.
SCENARIOS = (
    "toy_rumor_v0",
    "toy_provenance_stress_v0",
    "toy_adversarial_injection_v0",
)
TICKS = {
    "toy_rumor_v0": 10,
    "toy_provenance_stress_v0": 70,
    "toy_adversarial_injection_v0": 8,
}
SEEDS = (0, 1, 2)

# (label, backend, model_id)
MODELS: list[tuple[str, str, str]] = [
    ("gpt-4o", "openai", "gpt-4o"),
    ("qwen2.5-7b", "ollama", "qwen2.5:7b-instruct"),
]

METRIC_KEYS = (
    "gossip_count",
    "llm_invalid_action_rate",
    "belief_accuracy",
    "belief_brier",
    "belief_spread_fraction",
)


def _client(backend: str, model: str):
    if backend == "openai":
        os.environ["SBT_OPENAI_MODEL"] = model
        return OpenAIClient.from_env()
    if backend == "ollama":
        os.environ["SBT_OLLAMA_MODEL"] = model
        os.environ.setdefault("SBT_OLLAMA_TIMEOUT_S", "600")
        return OllamaClient.from_env()
    raise ValueError(backend)


def _metrics_path(trace: Path) -> Path:
    return trace.with_name(trace.stem + ".metrics.json")


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    out_dir = root / "runs" / "llm_eval" / "multimodel_h"
    out_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []

    for label, backend, model in MODELS:
        client = None
        for sid in SCENARIOS:
            scenario = json.loads((root / "scenarios" / f"{sid}.json").read_text(encoding="utf-8"))
            ticks = TICKS.get(sid, 12)
            for seed in SEEDS:
                trace = out_dir / f"{sid}__{label}__seed{seed}.jsonl"
                metrics = _metrics_path(trace)
                if metrics.exists():
                    row = json.loads(metrics.read_text(encoding="utf-8"))
                    rows.append(row)
                    print(f"skip {label} {sid} seed={seed}", flush=True)
                    continue
                if client is None:
                    client = _client(backend, model)
                if trace.exists():
                    trace.unlink()
                summary = run_episode(
                    scenario=scenario,
                    ticks=ticks,
                    seed=seed,
                    trace_path=trace,
                    policy_mode="llm_openai" if backend == "openai" else "llm_local",
                    llm_client=client,
                )
                row = {
                    "model_label": label,
                    "backend": backend,
                    "model_id": model,
                    "scenario_id": sid,
                    "seed": seed,
                }
                for mk in METRIC_KEYS:
                    row[mk] = summary.get(mk)
                uptake = summary.get("belief_uptake")
                if isinstance(uptake, dict):
                    row["trust_conf_pearson_r"] = uptake.get("trust_conf_pearson_r")
                metrics.write_text(json.dumps(row, indent=2, sort_keys=True), encoding="utf-8")
                rows.append(row)
                print(
                    f"done {label} {sid} seed={seed} "
                    f"gossip={row.get('gossip_count')} invalid={row.get('llm_invalid_action_rate')} "
                    f"acc={row.get('belief_accuracy')} brier={row.get('belief_brier')}",
                    flush=True,
                )

    (out_dir / "summary_raw.json").write_text(json.dumps(rows, indent=2, sort_keys=True), encoding="utf-8")
    # Aggregate
    from collections import defaultdict

    buckets: dict[tuple[str, str], dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for r in rows:
        key = (str(r["model_label"]), str(r["scenario_id"]))
        for k in list(METRIC_KEYS) + ["trust_conf_pearson_r"]:
            v = r.get(k)
            if isinstance(v, (int, float)):
                buckets[key][k].append(float(v))
    agg = []
    md = [
        "# H-tier multimodel",
        "",
        "| model | scenario | gossip | invalid | accuracy | brier | r |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for (label, sid), vals in sorted(buckets.items()):
        row = {"model_label": label, "scenario_id": sid, "n_seeds": len(SEEDS)}
        for k, xs in vals.items():
            row[f"{k}_mean"] = sum(xs) / len(xs)
        agg.append(row)
        md.append(
            f"| {label} | {sid} | {row.get('gossip_count_mean')} | {row.get('llm_invalid_action_rate_mean')} | "
            f"{row.get('belief_accuracy_mean')} | {row.get('belief_brier_mean')} | {row.get('trust_conf_pearson_r_mean')} |"
        )
    (out_dir / "summary_agg.json").write_text(json.dumps(agg, indent=2, sort_keys=True), encoding="utf-8")
    (out_dir / "summary_agg.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print((out_dir / "summary_agg.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
