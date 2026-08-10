"""Minimal third-party-style policy for adapter and CI contract demos."""

from __future__ import annotations

from npc_agent_benchmark.baselines.agents import BaseAgent
from npc_agent_benchmark.models import Action, AgentState, ExplanationTrace


class ExternalStubAgent(BaseAgent):
    """Non-LLM toy policy: always rests (predictable traces for harness tests)."""

    def act(self, *, t: int, state: AgentState, observation: dict[str, object]) -> tuple[Action, ExplanationTrace]:
        return Action(kind="rest"), ExplanationTrace(
            factors={"stub_policy": 1.0},
            notes=["ExternalStubAgent: placeholder policy for adapter contract demos."],
        )
