from __future__ import annotations

import random
from dataclasses import dataclass

from ..models import Action, AgentState, ExplanationTrace, NeedName


@dataclass(frozen=True)
class CandidateAction:
    action: Action
    base_utility: float
    factors: dict[str, float]


class BaseAgent:
    def __init__(self, agent_id: str, *, seed: int):
        self.agent_id = agent_id
        self._rng = random.Random(seed)

    def act(self, *, t: int, state: AgentState, observation: dict[str, object]) -> tuple[Action, ExplanationTrace]:
        raise NotImplementedError


class RoutineBaselineAgent(BaseAgent):
    """
    Simple schedule: rest at night-ish ticks, eat when hungry, otherwise socialize.
    No memory or belief use.
    """

    def act(self, *, t: int, state: AgentState, observation: dict[str, object]) -> tuple[Action, ExplanationTrace]:
        hunger = state.needs["hunger"].value
        rest = state.needs["rest"].value

        # Interpret need value as "satisfaction"; low means urgent.
        if hunger < 0.35:
            return (
                Action(kind="eat"),
                ExplanationTrace(factors={"need:hunger": 1.0}, notes=["Hunger low → eat."]),
            )
        if rest < 0.35 or (t % 24) >= 20:
            return (
                Action(kind="rest"),
                ExplanationTrace(factors={"need:rest": 1.0}, notes=["Rest low or late tick → rest."]),
            )
        return (
            Action(kind="socialize"),
            ExplanationTrace(factors={"need:social": 0.5}, notes=["Default routine → socialize."]),
        )


class UtilityNoMemoryAgent(BaseAgent):
    """
    Utility scorer over a few candidate actions with controlled noise.
    Does not use beliefs or past events (stateless policy over current needs).
    """

    ACTION_IMPACTS = {
        "eat": {"need": "hunger", "utility": 1.0},
        "rest": {"need": "rest", "utility": 1.0},
        "seek_safety": {"need": "safety", "utility": 1.0},
        "socialize": {"need": "social", "utility": 0.7},
        "steal_food": {
            "need": "hunger",
            "utility": 1.5,
            "breaks_value": "honesty",
            "achieves_goal": "wealth"
        },
        "gossip": {"need": "social", "utility": 1.1},
        "help_stranger": {
            "need": "social",
            "utility": 0.55,
            "achieves_goal": "aid_stranger",
            "value_align": "compassion",
        },
    }

    def __init__(
        self,
        agent_id: str,
        *,
        seed: int,
        noise: float = 0.05,
        disabled_action_kinds: set[str] | None = None,
    ):
        super().__init__(agent_id, seed=seed)
        self._noise = noise
        self._disabled_action_kinds = disabled_action_kinds or set()

    def _need_urgency(self, state: AgentState, need: NeedName) -> float:
        return 1.0 - state.needs[need].value

    def act(self, *, t: int, state: AgentState, observation: dict[str, object]) -> tuple[Action, ExplanationTrace]:
        cands: list[CandidateAction] = []
        
        for act_kind, impact in self.ACTION_IMPACTS.items():
            if act_kind in self._disabled_action_kinds:
                continue
            need_name = impact["need"]
            need_urgency = self._need_urgency(state, need_name)
            
            base_score = need_urgency * impact["utility"]
            factors: dict[str, float] = {f"need:{need_name}": base_score}
            score = base_score
            
            # Add Goal bonus if action achieves a known goal
            if "achieves_goal" in impact:
                goal_id = impact["achieves_goal"]
                if goal_id in state.goals:
                    bonus = state.goals[goal_id].priority
                    score += bonus
                    factors[f"goal:{goal_id}"] = bonus
                    
            # Subtract Value penalty if action breaks a known value
            if "breaks_value" in impact:
                val_id = impact["breaks_value"]
                if val_id in state.values:
                    penalty = state.values[val_id].rigidity
                    score -= penalty
                    factors[f"value_break:{val_id}"] = -penalty

            if act_kind == "help_stranger":
                peers = observation.get("peer_ids", [])
                if not isinstance(peers, list):
                    peers = []
                for pid in peers:
                    if not isinstance(pid, str) or pid == self.agent_id:
                        continue
                    hs_score = score
                    hs_factors = dict(factors)
                    if "value_align" in impact:
                        val_id = impact["value_align"]
                        if val_id in state.values:
                            vbonus = state.values[val_id].rigidity * 0.4
                            hs_score += vbonus
                            hs_factors[f"value_align:{val_id}"] = vbonus
                    hunger_urg = self._need_urgency(state, "hunger")
                    cost = 0.5 * hunger_urg
                    hs_score -= cost
                    hs_factors["altruism_cost:hunger"] = -cost
                    tr = state.relationships.get(pid).trust if pid in state.relationships else 0.0
                    hs_score += 0.12 * float(tr)
                    hs_factors[f"trust:{pid}"] = 0.12 * float(tr)
                    cands.append(
                        CandidateAction(
                            action=Action(kind="help_stranger", target=pid),
                            base_utility=hs_score,
                            factors=hs_factors,
                        )
                    )
                continue

            if act_kind == "gossip":
                if not state.beliefs or not state.relationships:
                    continue
                for rel_id, rel in state.relationships.items():
                    if rel.trust >= 0.0:
                        for b_id, b in state.beliefs.items():
                            if b.confidence > 0.6:
                                cand_score = score + b.confidence * 0.2 + rel.trust * 0.2
                                cand_factors = dict(factors)
                                cand_factors["belief_confidence"] = b.confidence * 0.2
                                cand_factors["trust_in_target"] = rel.trust * 0.2
                                cands.append(
                                    CandidateAction(
                                        action=Action(kind="gossip", target=rel_id, payload={"subject": b.subject}),
                                        base_utility=cand_score,
                                        factors=cand_factors
                                    )
                                )
                continue

            cands.append(
                CandidateAction(
                    action=Action(kind=act_kind),
                    base_utility=score,
                    factors=factors
                )
            )

        scored: list[tuple[float, CandidateAction]] = []
        for c in cands:
            noise = self._rng.uniform(-self._noise, self._noise)
            scored.append((c.base_utility + noise, c))

        scored.sort(key=lambda x: x[0], reverse=True)
        best_score, best = scored[0]
        # Emit enough structure for score trace arithmetic validation:
        # - score should be reconstructible as sum(contribs) + noise
        # - include runner-independent information about the action kind chosen
        contrib_sum = sum(float(v) for v in best.factors.values())
        noise = float(best_score - contrib_sum)
        return best.action, ExplanationTrace(
            factors={
                **best.factors,
                "contrib_sum": contrib_sum,
                "noise": noise,
                "score": float(best_score),
            },
            notes=["Max utility (+noise)."],
        )


class ReferenceCognitiveAgent(UtilityNoMemoryAgent):
    """
    "Ceiling" baseline: uses the same utility backbone, but also incorporates
    belief-derived risk signals and is more eager to communicate high-confidence beliefs.

    Still symbolic + inspectable (non-LLM).
    """

    def _danger_signal(self, state: AgentState) -> float:
        danger = 0.0
        for b in state.beliefs.values():
            txt = f"{b.subject} {b.proposition}".lower()
            if any(k in txt for k in ("bandit", "poison", "danger", "threat")):
                danger = max(danger, float(b.confidence))
        return danger

    def act(self, *, t: int, state: AgentState, observation: dict[str, object]) -> tuple[Action, ExplanationTrace]:
        # Build candidates by calling parent scoring, but inject belief modifiers by
        # temporarily adding "virtual" factors via a second pass.
        danger = self._danger_signal(state)

        action, expl = super().act(t=t, state=state, observation=observation)

        # If danger is high, bias toward seeking safety (override only when close).
        # This makes the agent "use beliefs" without needing a complex world model.
        if danger >= 0.65 and action.kind in ("socialize", "gossip"):
            expl.factors["belief_danger"] = danger
            expl.notes.append("High danger belief → prioritize safety.")
            return Action(kind="seek_safety"), expl

        # If the agent has high-confidence beliefs and trusted peers, gossip more often.
        if action.kind == "socialize" and state.beliefs and state.relationships:
            best_b = max(state.beliefs.values(), key=lambda b: float(b.confidence))
            if float(best_b.confidence) >= 0.75:
                # pick most trusted peer
                peer = max(state.relationships.items(), key=lambda kv: float(kv[1].trust))[0]
                expl.factors["belief_to_share"] = float(best_b.confidence)
                expl.notes.append("High-confidence belief → gossip instead of idle socialize.")
                return Action(kind="gossip", target=peer, payload={"subject": best_b.subject}), expl

        return action, expl


class GossipDrivenAgent(UtilityNoMemoryAgent):
    """Utility backbone, but prefers to gossip when it has a strong belief.

    This is intentionally simple: it exists to create policy-divergent multi-hop
    misinformation behavior without changing the shared belief-update harness.
    """

    def _most_trusted_peer(self, state: AgentState) -> str | None:
        if not state.relationships:
            return None
        peer, rel = max(state.relationships.items(), key=lambda item: float(item[1].trust))
        return peer if float(rel.trust) >= 0.0 else None

    def act(self, *, t: int, state: AgentState, observation: dict[str, object]) -> tuple[Action, ExplanationTrace]:
        if state.beliefs:
            best_b = max(state.beliefs.values(), key=lambda b: float(b.confidence))
            if float(best_b.confidence) >= 0.6:
                peer = self._most_trusted_peer(state)
                if peer is not None:
                    return (
                        Action(kind="gossip", target=peer, payload={"subject": best_b.subject}),
                        ExplanationTrace(
                            factors={"belief_confidence": float(best_b.confidence), f"trust:{peer}": float(state.relationships[peer].trust)},
                            notes=["Gossip-driven: share strongest belief with most trusted peer."],
                        ),
                    )

        return super().act(t=t, state=state, observation=observation)


class UtilityNoGossipAgent(UtilityNoMemoryAgent):
    """Utility policy with gossip permanently disabled (even when rumor channel is enabled)."""

    def __init__(
        self,
        agent_id: str,
        *,
        seed: int,
        noise: float = 0.05,
        disabled_action_kinds: set[str] | None = None,
    ):
        disabled = set(disabled_action_kinds or set())
        disabled.add("gossip")
        super().__init__(agent_id, seed=seed, noise=noise, disabled_action_kinds=disabled)


class ReputationProbeAgent(BaseAgent):
    """
    Independent symbolic probe policy inspired by reputation/trust systems.

    It does not reuse the utility scorer. It prioritizes high-confidence danger beliefs,
    avoids acting against low-trust peers, and shares high-confidence beliefs with the
    most trusted available peer.
    """

    def _strongest_belief(self, state: AgentState):
        if not state.beliefs:
            return None
        return max(state.beliefs.values(), key=lambda b: float(b.confidence))

    def _most_trusted_peer(self, state: AgentState) -> str | None:
        if not state.relationships:
            return None
        peer, rel = max(state.relationships.items(), key=lambda item: float(item[1].trust))
        return peer if float(rel.trust) >= 0.0 else None

    def act(self, *, t: int, state: AgentState, observation: dict[str, object]) -> tuple[Action, ExplanationTrace]:
        belief = self._strongest_belief(state)
        if belief is not None:
            text = f"{belief.subject} {belief.proposition}".lower()
            if float(belief.confidence) >= 0.65 and any(k in text for k in ("danger", "poison", "bandit", "threat")):
                return (
                    Action(kind="seek_safety"),
                    ExplanationTrace(
                        factors={"belief_risk": float(belief.confidence)},
                        notes=["Reputation probe: high-confidence danger belief."],
                    ),
                )

            peer = self._most_trusted_peer(state)
            if peer is not None and float(belief.confidence) >= 0.75:
                return (
                    Action(kind="gossip", target=peer, payload={"subject": belief.subject}),
                    ExplanationTrace(
                        factors={
                            "belief_confidence": float(belief.confidence),
                            f"trust:{peer}": float(state.relationships[peer].trust),
                        },
                        notes=["Reputation probe: share strong belief with trusted peer."],
                    ),
                )

        hunger = float(state.needs["hunger"].value)
        rest = float(state.needs["rest"].value)
        safety = float(state.needs["safety"].value)
        social = float(state.needs["social"].value)

        if safety < 0.45:
            return Action(kind="seek_safety"), ExplanationTrace(factors={"need:safety": 1.0 - safety})
        if hunger < 0.4:
            return Action(kind="eat"), ExplanationTrace(factors={"need:hunger": 1.0 - hunger})
        if rest < 0.35:
            return Action(kind="rest"), ExplanationTrace(factors={"need:rest": 1.0 - rest})
        if social < 0.45:
            return Action(kind="socialize"), ExplanationTrace(factors={"need:social": 1.0 - social})
        return Action(kind="rest"), ExplanationTrace(factors={"routine:rest": 0.1})

