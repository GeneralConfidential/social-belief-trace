"""Central registry for ``policy_mode`` → agent construction (single place to add policies)."""

from __future__ import annotations

from npc_agent_benchmark.baselines.agents import (
    BaseAgent,
    GossipDrivenAgent,
    ReputationProbeAgent,
    ReferenceCognitiveAgent,
    RoutineBaselineAgent,
    UtilityNoGossipAgent,
    UtilityNoMemoryAgent,
)
from npc_agent_benchmark.baselines.external_stub import ExternalStubAgent

# Modes accepted by ``run_episode`` (excluding legacy ``mixed``, which is handled separately).
REGISTERED_POLICY_MODES: frozenset[str] = frozenset(
    {
        "routine_only",
        "utility_only",
        "utility_no_gossip",
        "gossip_driven",
        "reference",
        "reputation_probe",
        "external_stub",
    }
)


def _disabled_actions(*, enable_rumor_channel: bool) -> set[str]:
    disabled: set[str] = set()
    if not enable_rumor_channel:
        disabled.add("gossip")
    return disabled


def build_agent(
    policy_mode: str,
    *,
    agent_id: str,
    index: int,
    seed: int,
    enable_rumor_channel: bool,
) -> BaseAgent:
    """Construct one agent policy. ``index`` disambiguates per-agent RNG streams."""
    disabled = _disabled_actions(enable_rumor_channel=enable_rumor_channel)
    if policy_mode == "routine_only":
        return RoutineBaselineAgent(agent_id, seed=seed + 100 + index)
    if policy_mode == "utility_only":
        return UtilityNoMemoryAgent(agent_id, seed=seed + 200 + index, disabled_action_kinds=disabled)
    if policy_mode == "utility_no_gossip":
        return UtilityNoGossipAgent(agent_id, seed=seed + 250 + index, disabled_action_kinds=disabled)
    if policy_mode == "gossip_driven":
        return GossipDrivenAgent(agent_id, seed=seed + 260 + index, disabled_action_kinds=disabled)
    if policy_mode == "reference":
        return ReferenceCognitiveAgent(agent_id, seed=seed + 300 + index, disabled_action_kinds=disabled)
    if policy_mode == "reputation_probe":
        return ReputationProbeAgent(agent_id, seed=seed + 400 + index)
    if policy_mode == "external_stub":
        return ExternalStubAgent(agent_id, seed=seed + 500 + index)
    raise ValueError(f"Unknown policy_mode: {policy_mode!r}")


def build_policies(
    *,
    agent_ids: list[str],
    policy_mode: str,
    seed: int,
    enable_rumor_channel: bool,
) -> dict[str, BaseAgent]:
    """Build the per-agent policy map for an episode."""
    policies: dict[str, BaseAgent] = {}
    if policy_mode == "mixed":
        for i, aid in enumerate(agent_ids):
            disabled = _disabled_actions(enable_rumor_channel=enable_rumor_channel)
            if i == 0:
                policies[aid] = RoutineBaselineAgent(aid, seed=seed + 100 + i)
            else:
                policies[aid] = UtilityNoMemoryAgent(aid, seed=seed + 200 + i, disabled_action_kinds=disabled)
        return policies

    if policy_mode not in REGISTERED_POLICY_MODES:
        raise ValueError(
            f"Unknown policy_mode: {policy_mode!r}. "
            f"Expected one of {sorted(REGISTERED_POLICY_MODES | frozenset({'mixed'}))}."
        )

    for i, aid in enumerate(agent_ids):
        policies[aid] = build_agent(
            policy_mode,
            agent_id=aid,
            index=i,
            seed=seed,
            enable_rumor_channel=enable_rumor_channel,
        )
    return policies
