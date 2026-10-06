"""Prompt construction for constrained LLM action selection."""

from __future__ import annotations

import json

from npc_agent_benchmark.baselines.action_schema import allowed_action_kinds
from npc_agent_benchmark.models import AgentState

PROMPT_VERSION = "llm_act_v2"
# Minor sensitivity variant: same schema, need-first priority when peer_ids empty or needs low.
PROMPT_VERSION_NEEDFIRST = "llm_act_v2_needfirst"


def build_act_prompt(
    *,
    agent_id: str,
    t: int,
    state: AgentState,
    observation: dict[str, object],
    disabled_action_kinds: set[str],
    prompt_version: str = PROMPT_VERSION,
) -> str:
    allowed = sorted(allowed_action_kinds(disabled_action_kinds=disabled_action_kinds))
    peer_ids = observation.get("peer_ids", [])
    if not isinstance(peer_ids, list):
        peer_ids = []
    peer_ids = [p for p in peer_ids if isinstance(p, str)]

    needs = observation.get("needs", {})
    if not isinstance(needs, dict):
        needs = {}

    messages = observation.get("messages", [])
    if not isinstance(messages, list):
        messages = []

    beliefs = sorted(state.beliefs.values(), key=lambda b: float(b.confidence), reverse=True)[:5]
    belief_rows = [
        {
            "subject": b.subject,
            "proposition": b.proposition,
            "source": b.source,
            "confidence": float(b.confidence),
        }
        for b in beliefs
    ]

    trust_rows = {
        peer: float(state.relationships[peer].trust)
        for peer in peer_ids
        if peer in state.relationships
    }

    context = {
        "prompt_version": prompt_version,
        "agent_id": agent_id,
        "tick": t,
        "needs": needs,
        "peer_ids": peer_ids,
        "messages_this_tick": messages,
        "top_beliefs": belief_rows,
        "trust_toward_peers": trust_rows,
        "allowed_action_kinds": allowed,
    }

    example_peer = peer_ids[0] if peer_ids else "peer_id"
    examples = [
        '{"kind":"rest","target":null,"payload":{}}',
        '{"kind":"eat","target":null,"payload":{}}',
        f'{{"kind":"gossip","target":"{example_peer}","payload":{{"subject":"food_shortage"}}}}',
        f'{{"kind":"help_stranger","target":"{example_peer}","payload":{{}}}}',
    ]

    priority = ""
    if prompt_version == PROMPT_VERSION_NEEDFIRST:
        priority = (
            "Priority (need-first variant):\n"
            "- If peer_ids is empty, never choose gossip or help_stranger; pick eat/rest/seek_safety/socialize/steal_food.\n"
            "- If hunger < 0.4 prefer eat; if rest < 0.4 prefer rest; if safety < 0.4 prefer seek_safety.\n"
            "- Only gossip when peer_ids is non-empty and you hold a belief with confidence >= 0.7.\n\n"
        )

    return (
        "You are a social agent policy inside a benchmark harness.\n"
        "Choose exactly one action for this tick.\n"
        "Output MUST be a single JSON object with ALL three keys: kind, target, payload.\n"
        "Do not omit kind. Do not wrap in markdown. No commentary.\n\n"
        "Rules:\n"
        f"- kind must be exactly one of: {allowed}\n"
        "- gossip: target = peer_id from peer_ids; payload = {\"subject\": \"<belief subject string>\"}\n"
        "- help_stranger: target = peer_id from peer_ids; payload = {}\n"
        "- all other kinds: target = null; payload = {}\n\n"
        + priority
        + "Valid examples:\n"
        + "\n".join(f"- {ex}" for ex in examples)
        + "\n\n"
        f"Context JSON:\n{json.dumps(context, indent=2, sort_keys=True)}\n\n"
        'Return only JSON like: {"kind":"...","target":null,"payload":{}}\n'
        "Action JSON:"
    )


def build_repair_prompt(*, previous_text: str, error: str, allowed_kinds: list[str], peer_ids: list[str]) -> str:
    example_peer = peer_ids[0] if peer_ids else "peer_id"
    return (
        "Your previous action JSON was invalid for this harness.\n"
        f"Error: {error}\n"
        f"Previous output:\n{previous_text[:800]}\n\n"
        f"allowed_action_kinds: {allowed_kinds}\n"
        f"peer_ids: {peer_ids}\n"
        "Return ONLY one corrected JSON object with keys kind, target, payload.\n"
        f'Example: {{"kind":"rest","target":null,"payload":{{}}}}\n'
        f'Example gossip: {{"kind":"gossip","target":"{example_peer}","payload":{{"subject":"rumor"}}}}\n'
        "Action JSON:"
    )
