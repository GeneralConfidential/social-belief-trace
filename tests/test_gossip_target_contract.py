"""Matched gossip-target contract (Cell B) vs legacy relationship targeting (Cell A)."""

from __future__ import annotations

import json
from pathlib import Path

from npc_agent_benchmark.baselines.agents import UtilityNoMemoryAgent, _eligible_gossip_targets
from npc_agent_benchmark.harness.runner import run_episode
from npc_agent_benchmark.models import AgentState, Belief, NeedState, Relationship


def _minimal_state() -> AgentState:
    return AgentState(
        agent_id="a",
        needs={
            "hunger": NeedState(value=0.8),
            "rest": NeedState(value=0.8),
            "safety": NeedState(value=0.8),
            "social": NeedState(value=0.2),
        },
        relationships={
            "s_p90": Relationship(trust=0.9),
            "peer": Relationship(trust=0.5),
        },
        beliefs={
            "b0": Belief(
                subject="r0",
                proposition="claim",
                source="s_p90",
                confidence=0.9,
                truth=True,
            )
        },
        goals={},
        values={},
    )


def test_eligible_targets_legacy_includes_sources() -> None:
    state = _minimal_state()
    obs = {"peer_ids": ["peer"]}
    got = set(_eligible_gossip_targets(state, obs))
    assert got == {"s_p90", "peer"}


def test_eligible_targets_peer_ids_contract() -> None:
    state = _minimal_state()
    obs = {"peer_ids": ["peer"], "gossip_target_contract": "peer_ids"}
    got = set(_eligible_gossip_targets(state, obs))
    assert got == {"peer"}


def test_utility_respects_peer_ids_contract() -> None:
    state = _minimal_state()
    agent = UtilityNoMemoryAgent("a", seed=0, noise=0.0)
    obs = {
        "t": 0,
        "peer_ids": ["peer"],
        "gossip_target_contract": "peer_ids",
        "messages": [],
    }
    # Force social urgency high; with contract, gossip target must be peer.
    action, _ = agent.act(t=0, state=state, observation=obs)
    if action.kind == "gossip":
        assert action.target == "peer"


def test_matched_scenario_episode_smoke(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    scenario = json.loads(
        (root / "scenarios" / "toy_provenance_stress_matched_gossip_v0.json").read_text(encoding="utf-8")
    )
    trace = tmp_path / "matched.jsonl"
    summary = run_episode(
        scenario=scenario,
        ticks=8,
        seed=0,
        trace_path=trace,
        policy_mode="utility_only",
    )
    assert summary.get("gossip_target_contract") == "peer_ids"
    assert summary.get("focal_agent_id") == "a"
    assert "peer" in summary.get("agents", [])

    # No gossip to source ids under the contract.
    for line in trace.read_text(encoding="utf-8").splitlines():
        ev = json.loads(line)
        if ev["agent_id"] != "a":
            assert ev["action"]["kind"] == "rest"
            continue
        if ev["action"]["kind"] == "gossip":
            assert ev["action"]["target"] == "peer"
            assert "peer" in (ev["observation"].get("peer_ids") or [])
