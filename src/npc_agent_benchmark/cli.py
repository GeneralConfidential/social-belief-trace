from __future__ import annotations

import argparse
import json
from pathlib import Path

from .harness.runner import run_episode


from npc_agent_benchmark.harness.policy_registry import REGISTERED_POLICY_MODES


def main() -> None:
    parser = argparse.ArgumentParser(prog="npc-agent-benchmark")
    parser.add_argument("--scenario", type=str, required=True, help="Path to scenario JSON file.")
    parser.add_argument(
        "--out",
        type=str,
        required=True,
        help="Output JSONL trace path (created/overwritten).",
    )
    parser.add_argument("--ticks", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    cli_choices = sorted(REGISTERED_POLICY_MODES | {"mixed"})
    parser.add_argument(
        "--policy-mode",
        default="utility_only",
        choices=cli_choices,
        help="Which policy family constructs all agents (default: utility_only).",
    )
    args = parser.parse_args()

    scenario_path = Path(args.scenario)
    out_path = Path(args.out)

    scenario = json.loads(scenario_path.read_text(encoding="utf-8"))
    summary = run_episode(
        scenario=scenario,
        ticks=args.ticks,
        seed=args.seed,
        trace_path=out_path,
        policy_mode=args.policy_mode,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))

