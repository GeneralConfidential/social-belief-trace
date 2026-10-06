"""Multi-model RQ6 slice for contrast table (OpenAI + Ollama backends)."""

from __future__ import annotations

import json
import os
from pathlib import Path

from npc_agent_benchmark.baselines.llm_client import OllamaClient, OpenAIClient
from npc_agent_benchmark.harness.runner import run_episode

SCENARIOS = (
    "toy_rumor_v0",
    "toy_provenance_stress_v0",
    "pubhealth_conflicting_guidance_v0",
    "toy_adversarial_injection_v0",
)
TICKS = {
    "toy_rumor_v0": 10,
    "toy_provenance_stress_v0": 70,
    "pubhealth_conflicting_guidance_v0": 12,
    "toy_adversarial_injection_v0": 8,
}
SEEDS = (0, 1, 2)

# (label, backend, model_id)
MODELS: list[tuple[str, str, str]] = [
    ("gpt-4o-mini", "openai", "gpt-4o-mini"),
    ("llama3.2-3b", "ollama", "llama3.2:3b-instruct-q4_K_M"),
    ("qwen2.5-3b", "ollama", "qwen2.5:3b-instruct"),
]

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

# Nested under belief_uptake in summary
UPTAKE_KEYS = ("trust_conf_pearson_r", "trust_conf_pearson_r_n")


def _client(backend: str, model: str):
    if backend == "openai":
        os.environ["SBT_OPENAI_MODEL"] = model
        return OpenAIClient.from_env()
    if backend == "ollama":
        os.environ["SBT_OLLAMA_MODEL"] = model
        return OllamaClient.from_env()
    raise ValueError(backend)


def _metrics_path(trace: Path) -> Path:
    return trace.with_name(trace.stem + ".metrics.json")


def _row_from_summary(
    *,
    label: str,
    backend: str,
    model: str,
    sid: str,
    seed: int,
    summary: dict[str, object],
) -> dict[str, object]:
    row: dict[str, object] = {
        "model_label": label,
        "backend": backend,
        "model_id": model,
        "scenario_id": sid,
        "seed": seed,
        "action_kind_counts": summary.get("action_kind_counts"),
    }
    for mk in METRIC_KEYS:
        row[mk] = summary.get(mk)
    uptake = summary.get("belief_uptake")
    if isinstance(uptake, dict):
        for uk in UPTAKE_KEYS:
            row[uk] = uptake.get(uk)
    else:
        for uk in UPTAKE_KEYS:
            row[uk] = summary.get(uk)
    return row


def _write_aggregates(out_dir: Path, rows: list[dict[str, object]]) -> None:
    raw_path = out_dir / "summary_raw.json"
    raw_path.write_text(json.dumps(rows, indent=2, sort_keys=True), encoding="utf-8")

    buckets: dict[tuple[str, str], dict[str, list[float]]] = {}
    keys = list(METRIC_KEYS) + list(UPTAKE_KEYS)
    for row in rows:
        key = (str(row["model_label"]), str(row["scenario_id"]))
        bucket = buckets.setdefault(key, {k: [] for k in keys})
        for k in keys:
            val = row.get(k)
            if isinstance(val, (int, float)):
                bucket[k].append(float(val))

    agg = []
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

    (out_dir / "summary_agg.json").write_text(json.dumps(agg, indent=2, sort_keys=True), encoding="utf-8")

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
    md_path = out_dir / "summary_agg.md"
    md_path.write_text("\n".join(md) + "\n", encoding="utf-8")
    print(md_path.read_text(encoding="utf-8"), flush=True)


def main() -> None:
    # Local model load + first JSON generate can exceed 120–300s on cold start.
    os.environ["SBT_OLLAMA_TIMEOUT_S"] = os.environ.get("SBT_OLLAMA_TIMEOUT_S", "600")

    root = Path(__file__).resolve().parents[1]
    out_dir = root / "runs" / "llm_eval" / "multimodel"
    out_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []

    for label, backend, model in MODELS:
        client = None
        for sid in SCENARIOS:
            scenario_path = root / "scenarios" / f"{sid}.json"
            if not scenario_path.exists():
                print(f"skip missing scenario {sid}", flush=True)
                continue
            scenario = json.loads(scenario_path.read_text(encoding="utf-8"))
            ticks = TICKS.get(sid, 12)
            for seed in SEEDS:
                trace = out_dir / f"{sid}__{label}__seed{seed}.jsonl"
                metrics = _metrics_path(trace)
                if metrics.exists():
                    row = json.loads(metrics.read_text(encoding="utf-8"))
                    rows.append(row)
                    print(
                        f"skip {label} {sid} seed={seed} "
                        f"gossip={row.get('gossip_count')} invalid={row.get('llm_invalid_action_rate')}",
                        flush=True,
                    )
                    continue
                if client is None:
                    client = _client(backend, model)
                summary = None
                last_err: Exception | None = None
                for attempt in range(1, 4):
                    try:
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
                        break
                    except (TimeoutError, RuntimeError) as exc:
                        msg = str(exc).lower()
                        if not isinstance(exc, TimeoutError) and "timed out" not in msg:
                            raise
                        last_err = exc
                        print(
                            f"timeout {label} {sid} seed={seed} attempt={attempt}/3; retrying",
                            flush=True,
                        )
                        # Recreate client in case the prior HTTP session is wedged.
                        client = _client(backend, model)
                if summary is None:
                    raise RuntimeError(
                        f"failed after retries: {label} {sid} seed={seed}"
                    ) from last_err
                row = _row_from_summary(
                    label=label,
                    backend=backend,
                    model=model,
                    sid=sid,
                    seed=seed,
                    summary=summary,
                )
                metrics.write_text(json.dumps(row, indent=2, sort_keys=True), encoding="utf-8")
                rows.append(row)
                print(
                    f"done {label} {sid} seed={seed} "
                    f"gossip={row.get('gossip_count')} invalid={row.get('llm_invalid_action_rate')} "
                    f"r={row.get('trust_conf_pearson_r')}",
                    flush=True,
                )

    _write_aggregates(out_dir, rows)


if __name__ == "__main__":
    main()
