"""Opt-in LLM evaluation (not part of default ``regenerate.py``).

From package root::

    uv run python scripts/run_llm_eval.py --backend mock
    uv run python scripts/run_llm_eval.py --backend ollama
    uv run python scripts/run_llm_eval.py --backend openai
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from npc_agent_benchmark.baselines.llm_client import MockLlmClient, OllamaClient, OpenAIClient
from npc_agent_benchmark.harness.runner import run_episode

LLM_EVAL_SCENARIOS: tuple[str, ...] = (
    "toy_rumor_v0",
    "toy_provenance_stress_v0",
    "pubhealth_conflicting_guidance_v0",
)

COMPARE_POLICY_MODES: tuple[str, ...] = (
    "llm_local",
    "utility_only",
    "reference",
)

LLM_METRIC_KEYS: tuple[str, ...] = (
    "gossip_count",
    "gossip_rate_per_event",
    "belief_spread_fraction",
    "help_stranger_rate_per_peer_event",
    "llm_invalid_action_rate",
    "llm_act_attempts",
    "llm_act_successes",
    "llm_act_fallbacks",
    "llm_act_repairs",
)

SEEDS: list[int] = [0, 1, 2, 3, 4]


def _load_ticks_by_scenario() -> dict[str, int]:
    import importlib.util

    agg_path = Path(__file__).resolve().parent / "run_all_agg.py"
    module_name = "_sbt_run_all_agg_ticks"
    spec = importlib.util.spec_from_file_location(module_name, agg_path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    ticks = getattr(mod, "DEFAULT_TICKS_BY_SCENARIO", {})
    return dict(ticks) if isinstance(ticks, dict) else {}


def _mean(xs: list[float]) -> float | None:
    return (sum(xs) / len(xs)) if xs else None


def _resolve_client(backend: str) -> object | None:
    if backend == "mock":
        return MockLlmClient(response='{"kind":"rest","target":null,"payload":{}}')
    if backend == "ollama":
        return OllamaClient.from_env()
    if backend == "openai":
        return OpenAIClient.from_env()
    raise ValueError(f"Unknown backend: {backend!r}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run opt-in LLM policy evaluation.")
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Alias for --backend mock (pipeline test, no network).",
    )
    parser.add_argument(
        "--backend",
        choices=("mock", "ollama", "openai"),
        default="mock",
        help="LLM backend (default: mock, no network).",
    )
    parser.add_argument(
        "--policy-modes",
        nargs="*",
        default=list(COMPARE_POLICY_MODES),
        help="Policy modes to run (default: llm_local utility_only reference).",
    )
    parser.add_argument(
        "--scenarios",
        nargs="*",
        default=list(LLM_EVAL_SCENARIOS),
        help="Scenario ids under scenarios/ (default: rumor, provenance_stress, pubhealth).",
    )
    parser.add_argument("--seeds", nargs="*", type=int, default=SEEDS)
    args = parser.parse_args()
    backend = "mock" if args.mock else args.backend

    root = Path(__file__).resolve().parents[1]
    scenarios_dir = root / "scenarios"
    runs_dir = root / "runs" / "llm_eval"
    runs_dir.mkdir(parents=True, exist_ok=True)

    client = _resolve_client(backend)
    raw_rows: list[dict[str, object]] = []
    ticks_by_scenario = _load_ticks_by_scenario()

    for sid in args.scenarios:
        scenario_path = scenarios_dir / f"{sid}.json"
        if not scenario_path.exists():
            raise FileNotFoundError(scenario_path)
        scenario = json.loads(scenario_path.read_text(encoding="utf-8"))
        ticks = int(ticks_by_scenario.get(sid, 24))

        for policy_mode in args.policy_modes:
            for seed in args.seeds:
                trace_name = f"{sid}__{policy_mode}__seed{seed}.jsonl"
                trace_path = runs_dir / trace_name
                use_client = client if policy_mode.startswith("llm_") else None
                summary = run_episode(
                    scenario=scenario,
                    ticks=ticks,
                    seed=seed,
                    trace_path=trace_path,
                    policy_mode=policy_mode,
                    llm_client=use_client,
                )
                summary["scenario_id"] = sid
                summary["policy_mode"] = policy_mode
                summary["seed"] = seed
                summary["backend"] = backend
                summary["trace_path"] = trace_path.relative_to(root).as_posix()
                raw_rows.append(summary)

    raw_path = runs_dir / f"summary_raw_{backend}.json"
    raw_path.write_text(json.dumps(raw_rows, indent=2, sort_keys=True), encoding="utf-8")
    # Legacy filenames for mock default path only (CI); named files preserve ollama vs openai.
    if backend == "mock":
        (runs_dir / "summary_raw.json").write_text(raw_path.read_text(encoding="utf-8"), encoding="utf-8")

    agg: dict[tuple[str, str], dict[str, list[float]]] = {}
    for row in raw_rows:
        key = (str(row["policy_mode"]), str(row["scenario_id"]))
        bucket = agg.setdefault(key, {mk: [] for mk in LLM_METRIC_KEYS})
        for mk in LLM_METRIC_KEYS:
            val = row.get(mk)
            if isinstance(val, (int, float)):
                bucket[mk].append(float(val))

    agg_rows: list[dict[str, object]] = []
    for (policy_mode, sid), vals in sorted(agg.items()):
        agg_row: dict[str, object] = {
            "policy_mode": policy_mode,
            "scenario_id": sid,
            "n_seeds": len(args.seeds),
            "backend": backend,
        }
        for mk in LLM_METRIC_KEYS:
            agg_row[f"{mk}_mean"] = _mean(vals.get(mk, []))
        agg_rows.append(agg_row)

    agg_path = runs_dir / f"summary_agg_{backend}.json"
    agg_path.write_text(json.dumps(agg_rows, indent=2, sort_keys=True), encoding="utf-8")
    if backend == "mock":
        (runs_dir / "summary_agg.json").write_text(agg_path.read_text(encoding="utf-8"), encoding="utf-8")

    md_cols = ["policy_mode", "scenario_id", "n_seeds", "backend"] + [f"{mk}_mean" for mk in LLM_METRIC_KEYS]
    lines = ["| " + " | ".join(md_cols) + " |", "| " + " | ".join(["---"] * len(md_cols)) + " |"]
    for row in agg_rows:
        cells = []
        for col in md_cols:
            val = row.get(col)
            if isinstance(val, float):
                cells.append(f"{val:.3f}")
            else:
                cells.append("" if val is None else str(val))
        lines.append("| " + " | ".join(cells) + " |")
    md_path = runs_dir / f"summary_agg_{backend}.md"
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    if backend == "mock":
        (runs_dir / "summary_agg.md").write_text(md_path.read_text(encoding="utf-8"), encoding="utf-8")

    print(f"Wrote {raw_path}")
    print(f"Wrote {agg_path}")
    print(f"Wrote {md_path}")
    print(md_path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
