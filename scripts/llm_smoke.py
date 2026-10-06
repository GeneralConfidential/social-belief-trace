"""Manual smoke: local Ollama LLM policy (requires Ollama running).

From ``social-belief-trace/``::

    ollama pull llama3.2:3b-instruct-q4_K_M
    uv run python scripts/llm_smoke.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from npc_agent_benchmark.harness.runner import run_episode


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    scenario_path = root / "scenarios" / "toy_scarcity_v0.json"
    if not scenario_path.exists():
        print(f"Missing scenario: {scenario_path}", file=sys.stderr)
        sys.exit(1)

    scenario = json.loads(scenario_path.read_text(encoding="utf-8"))
    out = root / "runs" / "llm_local_smoke.jsonl"
    summary = run_episode(
        scenario=scenario,
        ticks=5,
        seed=0,
        trace_path=out,
        policy_mode="llm_local",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
