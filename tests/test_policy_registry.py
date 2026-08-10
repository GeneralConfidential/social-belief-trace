from __future__ import annotations

import json
from pathlib import Path

import pytest

from npc_agent_benchmark.harness.policy_registry import REGISTERED_POLICY_MODES, build_agent, build_policies
from npc_agent_benchmark.models import TraceEvent


def test_registered_modes_cover_runner_surface() -> None:
    assert "external_stub" in REGISTERED_POLICY_MODES
    assert "utility_only" in REGISTERED_POLICY_MODES


def test_unknown_policy_raises() -> None:
    with pytest.raises(ValueError, match="Unknown policy_mode"):
        build_policies(agent_ids=["a"], policy_mode="not_a_real_mode", seed=0, enable_rumor_channel=True)


def test_mixed_policy_first_routine() -> None:
    policies = build_policies(agent_ids=["a", "b"], policy_mode="mixed", seed=1, enable_rumor_channel=True)
    assert policies["a"].__class__.__name__ == "RoutineBaselineAgent"
    assert policies["b"].__class__.__name__ == "UtilityNoMemoryAgent"


def test_external_stub_agent_constructible() -> None:
    agent = build_agent("external_stub", agent_id="x", index=0, seed=42, enable_rumor_channel=True)
    assert agent.__class__.__name__ == "ExternalStubAgent"


def test_trace_line_roundtrips_pydantic() -> None:
    root = Path(__file__).resolve().parents[1]
    sample = root / "runs" / "toy_scarcity_v0__utility_weighted_belief_update.jsonl"
    if not sample.exists():
        pytest.skip("runs/ not generated; run scripts/regenerate.py")
    line = sample.read_text(encoding="utf-8").splitlines()[0]
    ev = TraceEvent.model_validate_json(line)
    assert ev.trace_schema_version == "trace_v0"
    assert ev.t == 0


def test_schema_file_matches_model(tmp_path: Path) -> None:
    from npc_agent_benchmark.models import TraceEvent

    root = Path(__file__).resolve().parents[1]
    committed = root / "schemas" / "trace_v0.schema.json"
    if not committed.exists():
        pytest.skip("schema not committed")
    on_disk = json.loads(committed.read_text(encoding="utf-8"))
    fresh = TraceEvent.model_json_schema()
    assert on_disk == fresh
