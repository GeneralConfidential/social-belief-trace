from __future__ import annotations

import json
from pathlib import Path

from npc_agent_benchmark.harness.runner import run_episode


def test_external_stub_episode_writes_trace_v0(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    scenario = json.loads((root / "scenarios" / "toy_scarcity_v0.json").read_text(encoding="utf-8"))
    trace = tmp_path / "stub.jsonl"
    summary = run_episode(
        scenario=scenario,
        ticks=3,
        seed=0,
        trace_path=trace,
        policy_mode="external_stub",
    )
    assert summary.get("scenario_id") == "toy_scarcity_v0"
    lines = [ln for ln in trace.read_text(encoding="utf-8").splitlines() if ln.strip()]
    assert len(lines) == 6  # 3 ticks × 2 agents
    for ln in lines:
        obj = json.loads(ln)
        assert obj["trace_schema_version"] == "trace_v0"
        assert obj["action"]["kind"] == "rest"
