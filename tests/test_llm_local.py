from __future__ import annotations

import json

import pytest

from npc_agent_benchmark.baselines.llm_client import MockLlmClient
from npc_agent_benchmark.baselines.llm_local import (
    ConstrainedLlmAgent,
    extract_json_object,
    normalize_action_dict,
)
from npc_agent_benchmark.models import AgentState, NeedState


def _state() -> AgentState:
    return AgentState(
        agent_id="a",
        needs={
            "hunger": NeedState(value=0.5),
            "rest": NeedState(value=0.5),
            "safety": NeedState(value=0.5),
            "social": NeedState(value=0.5),
        },
    )


def test_extract_json_object_strips_fence() -> None:
    raw = 'Here you go:\n```json\n{"kind": "rest"}\n```'
    assert extract_json_object(raw) == {"kind": "rest"}


def test_extract_json_object_prefers_last_object() -> None:
    raw = 'prose {"kind":"eat","target":null,"payload":{}} trailing'
    assert extract_json_object(raw)["kind"] == "eat"


def test_normalize_infers_gossip_when_kind_missing() -> None:
    raw = {"target": "b", "payload": {"subject": "I am hungry"}}
    out = normalize_action_dict(raw, peer_ids=["b"])
    assert out["kind"] == "gossip"


def test_normalize_infers_rest_when_empty() -> None:
    out = normalize_action_dict({"target": None, "payload": {}}, peer_ids=["b"])
    assert out["kind"] == "rest"


def test_constrained_llm_agent_valid_json() -> None:
    client = MockLlmClient(response=json.dumps({"kind": "eat", "target": None, "payload": {}}))
    agent = ConstrainedLlmAgent("a", seed=0, client=client, model_id="mock", repair_on_failure=False)
    action, expl = agent.act(t=0, state=_state(), observation={"peer_ids": [], "needs": {}, "messages": []})
    assert action.kind == "eat"
    assert expl.factors.get("llm_valid") == 1.0
    stats = agent.llm_stats()
    assert stats["llm_act_successes"] == 1
    assert stats["llm_invalid_action_rate"] == 0.0
    assert stats["llm_prompt_version"] == "llm_act_v2"


def test_constrained_llm_agent_parse_failure_fallback() -> None:
    client = MockLlmClient(response="not json at all")
    agent = ConstrainedLlmAgent("a", seed=0, client=client, model_id="mock", repair_on_failure=False)
    action, expl = agent.act(t=0, state=_state(), observation={"peer_ids": [], "needs": {}, "messages": []})
    assert action.kind == "rest"
    assert expl.factors.get("llm_fallback") == 1.0
    stats = agent.llm_stats()
    assert stats["llm_act_parse_failures"] == 1
    assert stats["llm_act_fallbacks"] == 1
    assert any("raw=" in n for n in expl.notes)


def test_constrained_llm_agent_normalizes_missing_kind() -> None:
    client = MockLlmClient(response=json.dumps({"target": "b", "payload": {"subject": "food"}}))
    agent = ConstrainedLlmAgent("a", seed=0, client=client, model_id="mock", repair_on_failure=False)
    action, expl = agent.act(
        t=0,
        state=_state(),
        observation={"peer_ids": ["b"], "needs": {}, "messages": []},
    )
    assert action.kind == "gossip"
    assert action.target == "b"
    assert expl.factors.get("llm_valid") == 1.0


def test_constrained_llm_agent_repair_on_validation_failure() -> None:
    class SequencedClient:
        def __init__(self) -> None:
            self.n = 0

        def complete(self, prompt: str) -> str:
            self.n += 1
            if self.n == 1:
                return json.dumps({"kind": "fly", "target": None, "payload": {}})
            return json.dumps({"kind": "rest", "target": None, "payload": {}})

    agent = ConstrainedLlmAgent("a", seed=0, client=SequencedClient(), model_id="mock", repair_on_failure=True)
    action, expl = agent.act(t=0, state=_state(), observation={"peer_ids": [], "needs": {}, "messages": []})
    assert action.kind == "rest"
    assert expl.factors.get("llm_repaired") == 1.0
    assert agent.llm_stats()["llm_act_repairs"] == 1
