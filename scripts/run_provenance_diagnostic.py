"""Cell A / Cell B provenance-stress diagnostic (opt-in; not part of regenerate.py).

Cell A: auditable rerun of ``toy_provenance_stress_v0`` (empty peer_ids confound).
Cell B: matched gossip permissions via ``toy_provenance_stress_matched_gossip_v0``.

From package root::

    uv run python scripts/run_provenance_diagnostic.py --backend mock
    uv run python scripts/run_provenance_diagnostic.py --backend openai --policy-modes llm_openai utility_only reference
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from npc_agent_benchmark.baselines.llm_client import MockLlmClient, OllamaClient, OpenAIClient
from npc_agent_benchmark.harness.runner import run_episode

CELL_A_SCENARIO = "toy_provenance_stress_v0"
CELL_B_SCENARIO = "toy_provenance_stress_matched_gossip_v0"
TICKS = 70
SEEDS: list[int] = [0, 1, 2, 3, 4]

DEFAULT_POLICY_MODES: dict[str, tuple[str, ...]] = {
    "mock": ("llm_local", "utility_only", "reference"),
    "ollama": ("llm_local", "utility_only", "reference"),
    "openai": ("llm_openai", "utility_only", "reference"),
}

METRIC_KEYS: tuple[str, ...] = (
    "gossip_count",
    "gossip_rate_per_event",
    "belief_accuracy",
    "belief_brier",
    "llm_invalid_action_rate",
    "llm_act_attempts",
    "llm_act_successes",
    "llm_act_fallbacks",
    "llm_act_repairs",
)


def _mean(xs: list[float]) -> float | None:
    return (sum(xs) / len(xs)) if xs else None


def _resolve_client(backend: str) -> object | None:
    if backend == "mock":
        # Prefer gossip when peers exist so Cell B can exercise the matched path.
        return MockLlmClient(
            response='{"kind":"gossip","target":"peer","payload":{"subject":"r0"}}'
        )
    if backend == "ollama":
        return OllamaClient.from_env()
    if backend == "openai":
        return OpenAIClient.from_env()
    raise ValueError(f"Unknown backend: {backend!r}")


def _autopsy_row(path: Path, *, policy_mode: str, cell: str, seed: int) -> dict[str, object]:
    kinds: Counter[str] = Counter()
    peer_lens: Counter[int] = Counter()
    gossip_targets: Counter[str] = Counter()
    focal_gossip = 0
    fallback_notes = 0
    examples: list[dict[str, object]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        ev = json.loads(line)
        aid = str(ev.get("agent_id", ""))
        kind = str(ev.get("action", {}).get("kind", "?"))
        kinds[f"{aid}:{kind}"] += 1
        peers = ev.get("observation", {}).get("peer_ids") or []
        peer_lens[len(peers) if isinstance(peers, list) else -1] += 1
        notes = ev.get("explanation", {}).get("notes") or []
        if any("fallback" in str(n).lower() or "failure" in str(n).lower() for n in notes):
            fallback_notes += 1
        if kind == "gossip":
            target = ev.get("action", {}).get("target")
            gossip_targets[str(target)] += 1
            if aid == "a":
                focal_gossip += 1
        if aid == "a" and len(examples) < 3:
            examples.append(
                {
                    "t": ev.get("t"),
                    "action": ev.get("action"),
                    "peer_ids": peers,
                    "gossip_target_contract": ev.get("observation", {}).get("gossip_target_contract"),
                    "notes": notes[:3],
                }
            )
    return {
        "cell": cell,
        "policy_mode": policy_mode,
        "seed": seed,
        "trace": path.as_posix(),
        "action_kind_counts": dict(kinds),
        "peer_ids_len_counts": {str(k): v for k, v in sorted(peer_lens.items())},
        "gossip_targets": dict(gossip_targets),
        "focal_a_gossip_count": focal_gossip,
        "fallbackish_note_events": fallback_notes,
        "examples": examples,
    }


def _write_summaries(
    *,
    out_dir: Path,
    raw_rows: list[dict[str, object]],
    backend: str,
    seeds: list[int],
) -> None:
    raw_path = out_dir / "summary_raw.json"
    raw_path.write_text(json.dumps(raw_rows, indent=2, sort_keys=True), encoding="utf-8")

    agg: dict[tuple[str, str, str], dict[str, list[float]]] = {}
    for row in raw_rows:
        key = (str(row["cell"]), str(row["policy_mode"]), str(row["scenario_id"]))
        bucket = agg.setdefault(key, {mk: [] for mk in METRIC_KEYS})
        for mk in METRIC_KEYS:
            val = row.get(mk)
            if isinstance(val, (int, float)):
                bucket[mk].append(float(val))

    agg_rows: list[dict[str, object]] = []
    for (cell, policy_mode, sid), vals in sorted(agg.items()):
        agg_row: dict[str, object] = {
            "cell": cell,
            "policy_mode": policy_mode,
            "scenario_id": sid,
            "n_seeds": len(seeds),
            "backend": backend,
        }
        for mk in METRIC_KEYS:
            agg_row[f"{mk}_mean"] = _mean(vals.get(mk, []))
        agg_rows.append(agg_row)

    agg_path = out_dir / "summary_agg.json"
    agg_path.write_text(json.dumps(agg_rows, indent=2, sort_keys=True), encoding="utf-8")

    md_cols = ["cell", "policy_mode", "scenario_id", "n_seeds", "backend"] + [
        f"{mk}_mean" for mk in METRIC_KEYS
    ]
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
    md_path = out_dir / "summary.md"
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(md_path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Cell A/B provenance-stress diagnostic.")
    parser.add_argument(
        "--backend",
        choices=("mock", "ollama", "openai"),
        default="mock",
        help="LLM backend (default: mock, no network).",
    )
    parser.add_argument(
        "--policy-modes",
        nargs="*",
        default=None,
        help="Policy modes (default depends on backend).",
    )
    parser.add_argument("--seeds", nargs="*", type=int, default=SEEDS)
    parser.add_argument(
        "--cells",
        nargs="*",
        choices=("a", "b"),
        default=["a", "b"],
        help="Which cells to run (default: both).",
    )
    args = parser.parse_args()
    backend = args.backend
    policy_modes = tuple(args.policy_modes) if args.policy_modes else DEFAULT_POLICY_MODES[backend]
    seeds = list(args.seeds)

    root = Path(__file__).resolve().parents[1]
    scenarios_dir = root / "scenarios"
    base_out = root / "runs" / "llm_eval" / "provenance_diagnostic"
    base_out.mkdir(parents=True, exist_ok=True)

    client = _resolve_client(backend)
    cell_specs = {
        "a": (CELL_A_SCENARIO, "cell_a"),
        "b": (CELL_B_SCENARIO, "cell_b"),
    }

    raw_rows: list[dict[str, object]] = []
    autopsy_rows: list[dict[str, object]] = []

    for cell in args.cells:
        sid, cell_dir_name = cell_specs[cell]
        cell_dir = base_out / cell_dir_name
        cell_dir.mkdir(parents=True, exist_ok=True)
        scenario_path = scenarios_dir / f"{sid}.json"
        if not scenario_path.exists():
            raise FileNotFoundError(scenario_path)
        scenario = json.loads(scenario_path.read_text(encoding="utf-8"))

        for policy_mode in policy_modes:
            for seed in seeds:
                trace_name = f"{sid}__{policy_mode}__seed{seed}.jsonl"
                trace_path = cell_dir / trace_name
                use_client = client if policy_mode.startswith("llm_") else None
                summary = run_episode(
                    scenario=scenario,
                    ticks=TICKS,
                    seed=seed,
                    trace_path=trace_path,
                    policy_mode=policy_mode,
                    llm_client=use_client,
                )
                summary["cell"] = cell
                summary["scenario_id"] = sid
                summary["policy_mode"] = policy_mode
                summary["seed"] = seed
                summary["backend"] = backend
                summary["trace_path"] = trace_path.relative_to(root).as_posix()
                raw_rows.append(summary)
                if seed == seeds[0]:
                    autopsy_rows.append(
                        _autopsy_row(trace_path, policy_mode=policy_mode, cell=cell, seed=seed)
                    )

    _write_summaries(out_dir=base_out, raw_rows=raw_rows, backend=backend, seeds=seeds)

    finding = {
        "backend": backend,
        "policy_modes": list(policy_modes),
        "seeds": seeds,
        "finding": (
            "Cell A: single agent a => empty peer_ids; LLM gossip structurally unavailable; "
            "symbolic may still gossip to relationship/source ids (unequal permissions). "
            "Cell B: inert peer + gossip_target_contract=peer_ids + peer->external_stub; "
            "symbolic and LLM share eligible gossip targets; focal metrics on agent a."
        ),
        "rows": autopsy_rows,
    }
    autopsy_json = base_out / "autopsy.json"
    autopsy_json.write_text(json.dumps(finding, indent=2, sort_keys=True), encoding="utf-8")

    md_lines = [
        "# Provenance diagnostic autopsy (Cell A / Cell B)",
        "",
        finding["finding"],
        "",
        f"backend=`{backend}` policies=`{list(policy_modes)}` seeds=`{seeds}`",
        "",
    ]
    for row in autopsy_rows:
        md_lines.append(f"## Cell {row['cell'].upper()} — {row['policy_mode']} (seed {row['seed']})")
        md_lines.append(f"- focal_a_gossip_count: {row['focal_a_gossip_count']}")
        md_lines.append(f"- gossip_targets: `{row['gossip_targets']}`")
        md_lines.append(f"- peer_ids_len_counts: `{row['peer_ids_len_counts']}`")
        md_lines.append(f"- action_kind_counts: `{row['action_kind_counts']}`")
        md_lines.append(f"- fallbackish_note_events: {row['fallbackish_note_events']}")
        md_lines.append("- examples (agent a):")
        for ex in row["examples"]:
            md_lines.append(
                f"  - t={ex['t']} action={ex['action']} peer_ids={ex['peer_ids']} "
                f"contract={ex.get('gossip_target_contract')}"
            )
        md_lines.append("")
    autopsy_md = base_out / "autopsy.md"
    autopsy_md.write_text("\n".join(md_lines), encoding="utf-8")
    try:
        print(autopsy_md.read_text(encoding="utf-8"))
    except UnicodeEncodeError:
        print(autopsy_md.read_text(encoding="utf-8").encode("ascii", "replace").decode("ascii"))
    print(f"Wrote {base_out / 'summary_raw.json'}")
    print(f"Wrote {base_out / 'summary_agg.json'}")
    print(f"Wrote {autopsy_json}")


if __name__ == "__main__":
    main()
