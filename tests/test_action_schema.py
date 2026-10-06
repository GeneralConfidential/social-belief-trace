from __future__ import annotations

import pytest

from npc_agent_benchmark.baselines.action_schema import ActionValidationError, validate_action_dict
from npc_agent_benchmark.models import Action


def test_validate_simple_action() -> None:
    action = validate_action_dict({"kind": "eat", "target": None, "payload": {}}, peer_ids=[], disabled_action_kinds=set())
    assert action == Action(kind="eat")


def test_validate_gossip_requires_subject_and_peer() -> None:
    action = validate_action_dict(
        {"kind": "gossip", "target": "b", "payload": {"subject": "route"}},
        peer_ids=["a", "b"],
        disabled_action_kinds=set(),
    )
    assert action.kind == "gossip"
    assert action.target == "b"
    assert action.payload == {"subject": "route"}


def test_disabled_gossip_rejected() -> None:
    with pytest.raises(ActionValidationError, match="not allowed"):
        validate_action_dict(
            {"kind": "gossip", "target": "b", "payload": {"subject": "route"}},
            peer_ids=["b"],
            disabled_action_kinds={"gossip"},
        )
