from __future__ import annotations

import random
from pathlib import Path

from ..harness.policy_registry import build_policies
from ..metrics.v0 import summarize_v0
from ..models import Action, AgentState, Belief, ExplanationTrace, Goal, NeedState, Relationship, TraceEvent, Value

# Actions that are publicly observable (any bystander witnesses them and forms a belief).
_OBSERVABLE_ACTS: dict[str, dict[str, object]] = {
    "steal_food": {
        "proposition_template": "{actor} stole food",
        "truth": True,
        "violates_value": "honesty",
    },
}


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


def _clamp_minus1_plus1(x: float) -> float:
    return max(-1.0, min(1.0, x))


def _scenario_config(scenario: dict[str, object]) -> dict[str, object]:
    raw = scenario.get("config", {})
    return raw if isinstance(raw, dict) else {}


def _trust_to_weight(trust: float) -> float:
    # Map [-1, 1] -> [0, 1] (0 = fully discounted, 1 = fully trusted)
    return _clamp01(0.5 + 0.5 * trust)


def _deliver_messages(
    *,
    recipient: AgentState,
    messages: list[dict[str, object]],
    rumor_table: dict[str, dict[str, object]],
    use_provenance_weighting: bool,
    apply_belief_updates: bool,
) -> list[dict[str, object]]:
    delivered: list[dict[str, object]] = []
    for msg in messages:
        if msg.get("type") != "message":
            continue

        sender = msg.get("from")
        rumor_id = msg.get("rumor_id")
        if not isinstance(sender, str) or not isinstance(rumor_id, str):
            continue

        rumor = rumor_table.get(rumor_id)
        if not rumor:
            continue

        base_conf = float(msg.get("confidence", 0.5))
        trust = recipient.relationships.get(sender).trust if sender in recipient.relationships else 0.0
        if use_provenance_weighting:
            conf = _clamp01(base_conf * _trust_to_weight(float(trust)))
        else:
            # Ablation: ignore source trust/provenance when accepting a rumor.
            conf = _clamp01(base_conf)

        if apply_belief_updates:
            new_b = Belief(
                subject=str(rumor.get("subject", rumor_id)),
                proposition=str(rumor.get("proposition", "")),
                source=sender,
                confidence=conf,
                truth=rumor.get("truth"),
            )
            _reconcile_belief(recipient, new_b)
            # Social feedback: accepting a message nudges trust toward the sender (toy-scale).
            if sender not in recipient.relationships:
                recipient.relationships[sender] = Relationship(trust=0.0)
            bump = 0.04 * float(conf)
            recipient.relationships[sender].trust = _clamp_minus1_plus1(
                float(recipient.relationships[sender].trust) + bump
            )

        delivered.append(
            {
                "type": "message",
                "from": sender,
                "to": recipient.agent_id,
                "rumor_id": rumor_id,
                "base_confidence": base_conf,
                "trust": float(trust),
                "final_confidence": conf if apply_belief_updates else None,
                "belief_update_applied": apply_belief_updates,
            }
        )
    return delivered


def _init_state(agent_id: str, init: dict[str, object]) -> AgentState:
    needs_in = init.get("needs", {})
    needs = {
        "hunger": NeedState(value=float(needs_in.get("hunger", 0.8))),
        "rest": NeedState(value=float(needs_in.get("rest", 0.8))),
        "safety": NeedState(value=float(needs_in.get("safety", 0.8))),
        "social": NeedState(value=float(needs_in.get("social", 0.8))),
    }

    rels_in = init.get("relationships", {})
    relationships = {k: Relationship(trust=float(v.get("trust", 0.0))) for k, v in rels_in.items()}

    beliefs_in = init.get("beliefs", {})
    beliefs = {
        bid: Belief(
            subject=str(b.get("subject", bid)),
            proposition=str(b.get("proposition")),
            source=b.get("source"),
            confidence=float(b.get("confidence", 0.5)),
            truth=b.get("truth"),
        )
        for bid, b in beliefs_in.items()
    }

    goals_in = init.get("goals", {})
    goals = {
        gid: Goal(
            description=str(g.get("description", "")),
            priority=float(g.get("priority", 0.5)),
        )
        for gid, g in goals_in.items()
    }

    values_in = init.get("values", {})
    values = {
        vid: Value(
            description=str(v.get("description", "")),
            rigidity=float(v.get("rigidity", 0.5)),
        )
        for vid, v in values_in.items()
    }

    return AgentState(
        agent_id=agent_id,
        needs=needs,
        relationships=relationships,
        beliefs=beliefs,
        goals=goals,
        values=values,
    )


def _reconcile_belief(state: AgentState, new_belief: Belief) -> None:
    subject = new_belief.subject
    
    existing_key = None
    existing_b = None
    for k, b in state.beliefs.items():
        if b.subject == subject:
            existing_key = k
            existing_b = b
            break
            
    if not existing_b:
        state.beliefs[subject] = new_belief
        return
        
    if existing_b.truth == new_belief.truth:
        # Reinforcement: max confidence
        existing_b.confidence = max(existing_b.confidence, new_belief.confidence)
        # We can also update the source if the new source is "witnessed", maybe.
        return
        
    # Contradiction Detective
    if new_belief.confidence > existing_b.confidence + 0.1:
        # Overwrite with stronger conflicting info
        if existing_key != subject and existing_key is not None:
            del state.beliefs[existing_key]
        state.beliefs[subject] = new_belief
    elif abs(new_belief.confidence - existing_b.confidence) <= 0.1:
        # Too close to call: Mark as disputed by lowering confidence
        existing_b.confidence = _clamp01(existing_b.confidence - 0.2)
    else:
        # Ignore weaker info, but introduce slight doubt
        existing_b.confidence = _clamp01(existing_b.confidence - 0.05)


def _witness_actions(
    *,
    actor_id: str,
    action: Action,
    all_states: dict[str, AgentState],
) -> list[dict[str, object]]:
    """After an agent acts, broadcast observable actions to all other agents as direct beliefs.

    Confidence is boosted by how much the witness trusts the actor — direct observation
    is treated as a first-hand belief, so trust acts as a *certainty amplifier* here
    (you are more shocked/certain when you trust someone and they betray your expectations).
    """
    spec = _OBSERVABLE_ACTS.get(action.kind)
    if spec is None:
        return []

    witnessed: list[dict[str, object]] = []
    proposition = spec["proposition_template"].format(actor=actor_id)  # type: ignore[arg-type]
    truth: bool | None = spec.get("truth")  # type: ignore[assignment]

    for wid, wstate in all_states.items():
        if wid == actor_id:
            continue  # You cannot witness your own action as an outside observer

        # Base confidence = 0.8 (direct observation is high certainty).
        # Trust modulates it: high trust → slightly higher certainty (shocking clarity);
        # low trust → slight doubt ("did I really see that?").
        base_conf = 0.8
        trust = wstate.relationships.get(actor_id).trust if actor_id in wstate.relationships else 0.0
        # trust in [-1,1]: add ±0.15 amplitude
        conf = _clamp01(base_conf + 0.15 * float(trust))

        subject = f"witnessed:{actor_id}:{action.kind}"
        new_b = Belief(
            subject=subject,
            proposition=proposition,
            source="witnessed",
            confidence=conf,
            truth=truth,
        )
        _reconcile_belief(wstate, new_b)

        # NEW: Trust penalty logic
        violated_value = spec.get("violates_value")
        if isinstance(violated_value, str) and violated_value in wstate.values:
            rigidity = wstate.values[violated_value].rigidity
            # Scale penalty by rigidity: more rigid = bigger trust hit.
            # Base penalty of 0.25 for theft.
            penalty = 0.25 * rigidity
            
            if actor_id not in wstate.relationships:
                wstate.relationships[actor_id] = Relationship(trust=0.0)
            
            old_trust = wstate.relationships[actor_id].trust
            wstate.relationships[actor_id].trust = _clamp_minus1_plus1(old_trust - penalty)

        witnessed.append(
            {
                "type": "witnessed",
                "actor": actor_id,
                "action_kind": action.kind,
                "witness": wid,
                "trust_in_actor_before": float(trust),
                "trust_in_actor_after": float(wstate.relationships[actor_id].trust) if actor_id in wstate.relationships else 0.0,
                "belief_confidence": conf,
                "belief_id": subject,
            }
        )
    return witnessed


def _apply_belief_decay(state: AgentState, *, rng: random.Random, decay_multiplier: float = 1.0) -> None:
    """Gradually reduce confidence of all beliefs. Remove those that fall too low."""
    to_delete = []
    # Base decay rate of 0.005 per tick
    decay_rate = 0.005 * max(0.0, float(decay_multiplier))

    for bid, belief in state.beliefs.items():
        loss = rng.uniform(decay_rate * 0.5, decay_rate * 1.5)
        belief.confidence = _clamp01(belief.confidence - loss)

        # If confidence is negligible, mark for removal
        if belief.confidence < 0.05:
            to_delete.append(bid)

    for bid in to_delete:
        del state.beliefs[bid]


def _apply_action(
    state: AgentState,
    action: Action,
    all_states: dict[str, AgentState],
    *,
    rng: random.Random,
    apply_belief_updates: bool = True,
    need_passive_decay_multiplier: float = 1.0,
) -> None:
    # Very small toy dynamics for v0: actions nudge needs.
    if action.kind == "eat":
        state.needs["hunger"].value = min(1.0, state.needs["hunger"].value + 0.25)
        state.needs["safety"].value = max(0.0, state.needs["safety"].value - 0.01)
    elif action.kind == "rest":
        state.needs["rest"].value = min(1.0, state.needs["rest"].value + 0.25)
        state.needs["hunger"].value = max(0.0, state.needs["hunger"].value - 0.02)
    elif action.kind == "seek_safety":
        state.needs["safety"].value = min(1.0, state.needs["safety"].value + 0.2)
        state.needs["social"].value = max(0.0, state.needs["social"].value - 0.02)
    elif action.kind == "socialize":
        state.needs["social"].value = min(1.0, state.needs["social"].value + 0.2)
        state.needs["rest"].value = max(0.0, state.needs["rest"].value - 0.01)
    elif action.kind == "steal_food":
        # Massive jump in hunger satisfaction, but massive drop in social standing
        state.needs["hunger"].value = min(1.0, state.needs["hunger"].value + 0.5)
        state.needs["social"].value = max(0.0, state.needs["social"].value - 0.2)
    elif action.kind == "help_stranger":
        # Cost to helper: hunger/rest; benefit to target: hunger (+ small social for both).
        tid = action.target
        if isinstance(tid, str) and tid in all_states and tid != state.agent_id:
            state.needs["hunger"].value = max(0.0, state.needs["hunger"].value - 0.14)
            state.needs["rest"].value = max(0.0, state.needs["rest"].value - 0.05)
            state.needs["social"].value = min(1.0, state.needs["social"].value + 0.12)
            tgt = all_states[tid]
            tgt.needs["hunger"].value = min(1.0, tgt.needs["hunger"].value + 0.28)
            tgt.needs["social"].value = min(1.0, tgt.needs["social"].value + 0.08)
    elif action.kind == "gossip":
        state.needs["social"].value = min(1.0, state.needs["social"].value + 0.3)
        if not apply_belief_updates:
            return
        target_id = action.target
        subject = action.payload.get("subject")
        if isinstance(target_id, str) and isinstance(subject, str):
            target_state = all_states.get(target_id)
            if target_state:
                b = None
                for b_val in state.beliefs.values():
                    if b_val.subject == subject:
                        b = b_val
                        break
                if b is not None:
                    base_conf = b.confidence
                    trust = target_state.relationships.get(state.agent_id).trust if state.agent_id in target_state.relationships else 0.0
                    conf = _clamp01(base_conf * _trust_to_weight(float(trust)))
                    new_b = Belief(
                        subject=subject,
                        proposition=b.proposition,
                        source=state.agent_id,
                        confidence=conf,
                        truth=b.truth,
                    )
                    _reconcile_belief(target_state, new_b)

    # Passive decay each tick (keeps things moving).
    decay_scale = max(0.0, float(need_passive_decay_multiplier))
    for k in state.needs:
        state.needs[k].value = max(
            0.0,
            min(1.0, state.needs[k].value - rng.uniform(0.0, 0.015) * decay_scale),
        )


def run_episode(
    *,
    scenario: dict[str, object],
    ticks: int,
    seed: int,
    trace_path: Path,
    use_provenance_weighting: bool = True,
    enable_rumor_channel: bool = True,
    apply_belief_updates: bool = True,
    policy_mode: str = "utility_only",
) -> dict[str, object]:
    rng = random.Random(seed)

    agents_spec = scenario.get("agents", {})
    if not isinstance(agents_spec, dict) or not agents_spec:
        raise ValueError("Scenario must include non-empty object 'agents'.")

    rumor_table = scenario.get("rumors", {})
    rumor_table = rumor_table if isinstance(rumor_table, dict) else {}

    scheduled_events = scenario.get("events", [])
    scheduled_events = scheduled_events if isinstance(scheduled_events, list) else []
    cfg = _scenario_config(scenario)
    belief_decay_mult = float(cfg.get("belief_decay_multiplier", 1.0))
    need_decay_mult = float(cfg.get("need_passive_decay_multiplier", 1.0))
    primary_subject = cfg.get("primary_subject")
    primary_subject = str(primary_subject) if isinstance(primary_subject, str) else None

    states: dict[str, AgentState] = {aid: _init_state(aid, init) for aid, init in agents_spec.items()}
    # Scenario-initial relationship snapshot for trust-drift metrics.
    initial_relationship_trust: dict[str, dict[str, float]] = {
        aid: {rid: float(rel.trust) for rid, rel in st.relationships.items()} for aid, st in states.items()
    }

    # Policy selection: see ``policy_registry`` for supported ``policy_mode`` values.
    agent_ids = list(states.keys())
    policies = build_policies(
        agent_ids=agent_ids,
        policy_mode=policy_mode,
        seed=seed,
        enable_rumor_channel=enable_rumor_channel,
    )

    events: list[TraceEvent] = []
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    with trace_path.open("w", encoding="utf-8") as f:
        for t in range(ticks):
            # Gather messages scheduled for this tick
            msgs_by_recipient: dict[str, list[dict[str, object]]] = {}
            if enable_rumor_channel:
                for ev in scheduled_events:
                    if not isinstance(ev, dict):
                        continue
                    if int(ev.get("t", -1)) != t:
                        continue
                    if ev.get("type") != "message":
                        continue
                    to = ev.get("to")
                    if isinstance(to, str):
                        msgs_by_recipient.setdefault(to, []).append(ev)
            for rec in msgs_by_recipient:
                msgs_by_recipient[rec] = sorted(
                    msgs_by_recipient[rec],
                    key=lambda m: (
                        str(m.get("from", "")),
                        str(m.get("rumor_id", "")),
                        float(m.get("confidence", 0.0)),
                    ),
                )

            for aid in agent_ids:
                state = states[aid]
                delivered = []
                if enable_rumor_channel:
                    delivered = _deliver_messages(
                        recipient=state,
                        messages=msgs_by_recipient.get(aid, []),
                        rumor_table=rumor_table,
                        use_provenance_weighting=use_provenance_weighting,
                        apply_belief_updates=apply_belief_updates,
                    )
                peer_ids = [x for x in agent_ids if x != aid]
                obs = {
                    "t": t,
                    "needs": {k: float(v.value) for k, v in state.needs.items()},
                    "messages": delivered,
                    "peer_ids": peer_ids,
                }

                action, expl = policies[aid].act(t=t, state=state, observation=obs)
                if not isinstance(expl, ExplanationTrace):
                    expl = ExplanationTrace.model_validate(expl)

                if enable_rumor_channel:
                    _apply_action(
                        state,
                        action,
                        states,
                        rng=rng,
                        apply_belief_updates=apply_belief_updates,
                        need_passive_decay_multiplier=need_decay_mult,
                    )
                else:
                    # Rumor channel ablation: ignore gossip entirely (but keep other actions).
                    if action.kind != "gossip":
                        _apply_action(
                            state,
                            action,
                            states,
                            rng=rng,
                            apply_belief_updates=apply_belief_updates,
                            need_passive_decay_multiplier=need_decay_mult,
                        )

                if apply_belief_updates:
                    _apply_belief_decay(state, rng=rng, decay_multiplier=belief_decay_mult)

                # Broadcast observable actions → witnesses form beliefs immediately.
                witnessed = []
                if apply_belief_updates:
                    witnessed = _witness_actions(
                        actor_id=aid,
                        action=action,
                        all_states=states,
                    )
                if witnessed:
                    obs["witnessed_by_others"] = witnessed

                ev = TraceEvent(
                    t=t,
                    agent_id=aid,
                    observation=obs,
                    action=action,
                    explanation=expl,
                    state_after=state.model_copy(deep=True),
                )
                events.append(ev)
                f.write(ev.model_dump_json())
                f.write("\n")

    summary = summarize_v0(
        events,
        initial_relationship_trust=initial_relationship_trust,
        primary_subject=primary_subject,
    )
    summary["trace_path"] = str(trace_path)
    summary["scenario_id"] = scenario.get("id", "unknown")
    return summary

