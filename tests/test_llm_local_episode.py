from __future__ import annotations

import json
from pathlib import Path

from npc_agent_benchmark.baselines.llm_client import MockLlmClient
from npc_agent_benchmark.harness.runner import run_episode


def test_llm_local_episode_aggregates_stats(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    scenario_path = root / "scenarios" / "toy_scarcity_v0.json"
    if not scenario_path.exists():
        return

    scenario = json.loads(scenario_path.read_text(encoding="utf-8"))
    mock = MockLlmClient(response='{"kind":"rest","target":null,"payload":{}}')
    trace = tmp_path / "llm_local.jsonl"
    summary = run_episode(
        scenario=scenario,
        ticks=2,
        seed=0,
        trace_path=trace,
        policy_mode="llm_local",
        llm_client=mock,
    )
    assert summary.get("llm_act_attempts") == 4
    assert summary.get("llm_act_successes") == 4
    assert summary.get("llm_invalid_action_rate") == 0.0
    lines = [ln for ln in trace.read_text(encoding="utf-8").splitlines() if ln.strip()]
    assert len(lines) == 4
