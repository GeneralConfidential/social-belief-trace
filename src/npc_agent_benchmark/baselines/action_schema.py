"""Closed action vocabulary and validation for constrained LLM policies."""

from __future__ import annotations

from npc_agent_benchmark.models import Action

ALL_ACTION_KINDS: frozenset[str] = frozenset(
    {
        "eat",
        "rest",
        "seek_safety",
        "socialize",
        "steal_food",
        "gossip",
        "help_stranger",
    }
)

TARGET_REQUIRED_KINDS: frozenset[str] = frozenset({"gossip", "help_stranger"})


class ActionValidationError(ValueError):
    """Raised when a parsed action dict fails harness constraints."""


def allowed_action_kinds(*, disabled_action_kinds: set[str]) -> frozenset[str]:
    return ALL_ACTION_KINDS - frozenset(disabled_action_kinds)


def validate_action_dict(
    raw: object,
    *,
    peer_ids: list[str],
    disabled_action_kinds: set[str],
) -> Action:
    if not isinstance(raw, dict):
        raise ActionValidationError("action must be a JSON object")

    kind = raw.get("kind")
    if not isinstance(kind, str) or not kind:
        raise ActionValidationError("action.kind must be a non-empty string")

    allowed = allowed_action_kinds(disabled_action_kinds=disabled_action_kinds)
    if kind not in allowed:
        raise ActionValidationError(f"action.kind {kind!r} is not allowed this tick")

    target = raw.get("target")
    payload = raw.get("payload", {})

    if target is not None and not isinstance(target, str):
        raise ActionValidationError("action.target must be a string or null")

    if not isinstance(payload, dict):
        raise ActionValidationError("action.payload must be an object")

    peer_set = set(peer_ids)

    if kind in TARGET_REQUIRED_KINDS:
        if not isinstance(target, str) or target not in peer_set:
            raise ActionValidationError(f"action.kind {kind!r} requires target in peer_ids")
    elif target is not None:
        raise ActionValidationError(f"action.kind {kind!r} must not set target")

    if kind == "gossip":
        subject = payload.get("subject")
        if not isinstance(subject, str) or not subject.strip():
            raise ActionValidationError("gossip requires payload.subject string")
        return Action(kind=kind, target=target, payload={"subject": subject})

    if kind == "help_stranger":
        return Action(kind=kind, target=target, payload={})

    if payload:
        raise ActionValidationError(f"action.kind {kind!r} must use an empty payload")

    return Action(kind=kind, target=None, payload={})
