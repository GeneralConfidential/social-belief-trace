from __future__ import annotations

from collections import Counter
from math import sqrt

from ..models import TraceEvent


def _goal_proxy_action_kind(goal_id: str) -> str | None:
    # v0 proxies: these are intentionally simple and scenario-agnostic.
    return {
        "aid_stranger": "help_stranger",
        "wealth": "steal_food",
        # "survive" is handled via a need-based proxy below
    }.get(goal_id)


def _pearson(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) != len(ys) or len(xs) < 2:
        return None
    mx = sum(xs) / len(xs)
    my = sum(ys) / len(ys)
    dx = [x - mx for x in xs]
    dy = [y - my for y in ys]
    denom = sqrt(sum(a * a for a in dx) * sum(b * b for b in dy))
    if denom == 0.0:
        return None
    return sum(a * b for a, b in zip(dx, dy, strict=True)) / denom


def summarize_v0(
    events: list[TraceEvent],
    *,
    initial_relationship_trust: dict[str, dict[str, float]] | None = None,
    primary_subject: str | None = None,
) -> dict[str, object]:
    if not events:
        return {"error": "no_events"}

    # Need satisfaction: average of final needs across agents.
    final_by_agent: dict[str, TraceEvent] = {}
    for ev in events:
        final_by_agent[ev.agent_id] = ev

    avg_final_needs: dict[str, float] = {}
    n_agents = len(final_by_agent)
    if n_agents:
        totals: dict[str, float] = {}
        for ev in final_by_agent.values():
            for need, val in ev.state_after.needs.items():
                totals[need] = totals.get(need, 0.0) + float(val.value)
        avg_final_needs = {k: v / n_agents for k, v in totals.items()}

    # Trust drift: change from scenario-initial → final averaged over relationship keys.
    # If an initial snapshot isn't provided, fall back to the first state_after in the trace.
    initial_rel = initial_relationship_trust or {}
    fallback_initial_by_agent: dict[str, TraceEvent] = {}
    if not initial_rel:
        for ev in events:
            if ev.agent_id not in fallback_initial_by_agent:
                fallback_initial_by_agent[ev.agent_id] = ev

    trust_drift_abs: float = 0.0
    trust_drift_signed: float = 0.0
    trust_pairs: int = 0
    for aid, last in final_by_agent.items():
        init_map = initial_rel.get(aid, {})
        if not init_map and aid in fallback_initial_by_agent:
            init_map = {k: float(v.trust) for k, v in fallback_initial_by_agent[aid].state_after.relationships.items()}

        last_map = {k: float(v.trust) for k, v in last.state_after.relationships.items()}
        keys = set(init_map.keys()) | set(last_map.keys())
        for k in keys:
            v0 = float(init_map.get(k, 0.0))
            v1 = float(last_map.get(k, 0.0))
            trust_drift_abs += abs(v1 - v0)
            trust_drift_signed += v1 - v0
            trust_pairs += 1

    avg_trust_drift_abs = trust_drift_abs / trust_pairs if trust_pairs else 0.0
    avg_trust_drift_signed = trust_drift_signed / trust_pairs if trust_pairs else 0.0

    # Action diversity: entropy-ish proxy via unique action kinds.
    action_kinds = [ev.action.kind for ev in events]
    counts = Counter(action_kinds)
    unique_kinds = len(counts)

    # Belief metrics (if beliefs carry truth labels)
    belief_n = 0  # truth-labeled beliefs evaluated (final snapshot)
    belief_total_final = 0  # total beliefs present in final snapshot (all labels)
    belief_acc_n = 0
    brier_sum = 0.0
    uptake_pairs: list[tuple[float, float]] = []  # (trust, final_confidence)
    for ev in final_by_agent.values():
        belief_total_final += len(ev.state_after.beliefs)
        for b in ev.state_after.beliefs.values():
            if b.truth is None:
                continue
            belief_n += 1
            y = 1.0 if bool(b.truth) else 0.0
            p = float(b.confidence)
            brier_sum += (p - y) ** 2
            pred = p >= 0.5
            if pred == bool(b.truth):
                belief_acc_n += 1

    # Aggregate uptake statistics across all delivered messages in the trace.
    trust_samples: list[float] = []
    conf_samples: list[float] = []
    for ev in events:
        msgs = ev.observation.get("messages")
        if not isinstance(msgs, list):
            continue
        for m in msgs:
            if not isinstance(m, dict):
                continue
            trust = m.get("trust")
            fc = m.get("final_confidence")
            if isinstance(trust, (int, float)) and isinstance(fc, (int, float)):
                uptake_pairs.append((float(trust), float(fc)))
                trust_samples.append(float(trust))
                conf_samples.append(float(fc))

    belief_accuracy = (belief_acc_n / belief_n) if belief_n else None
    belief_brier = (brier_sum / belief_n) if belief_n else None
    belief_truth_labeled_fraction = (belief_n / belief_total_final) if belief_total_final else None

    # Simple uptake summary: average final confidence for negative vs non-negative trust.
    neg = [fc for tr, fc in uptake_pairs if tr < 0]
    pos = [fc for tr, fc in uptake_pairs if tr >= 0]
    n_messages = len(uptake_pairs)
    r_trust_conf = _pearson(trust_samples, conf_samples) if n_messages >= 5 else None
    uptake_summary = {
        "n_messages": n_messages,
        "avg_final_confidence_trust_lt_0": (sum(neg) / len(neg)) if neg else None,
        "avg_final_confidence_trust_gte_0": (sum(pos) / len(pos)) if pos else None,
        "trust_conf_pearson_r": r_trust_conf,
        "trust_conf_pearson_r_n": n_messages,
    }

    n_events = len(events)
    gossip_count = sum(1 for ev in events if ev.action.kind == "gossip")
    gossip_rate_per_event = gossip_count / n_events if n_events else None

    help_stranger_count = sum(1 for ev in events if ev.action.kind == "help_stranger")
    help_stranger_peer_events = sum(
        1
        for ev in events
        if isinstance(ev.observation.get("peer_ids"), list) and len(ev.observation.get("peer_ids", [])) > 0
    )
    # Belief spread / subject diversity (final snapshot).
    subjects_final: set[str] = set()
    agents_with_primary = 0
    for ev in final_by_agent.values():
        for b in ev.state_after.beliefs.values():
            if b.subject:
                subjects_final.add(str(b.subject))
            if primary_subject and b.subject == primary_subject and float(b.confidence) >= 0.15:
                agents_with_primary += 1
                break
    unique_belief_subjects_final = len(subjects_final)
    belief_spread_agents = agents_with_primary if primary_subject else None
    belief_spread_fraction = (
        agents_with_primary / n_agents if primary_subject and n_agents else None
    )
    value_break_count = sum(
        1
        for ev in events
        if any(str(k).startswith("value_break:") for k in ev.explanation.factors.keys())
    )
    help_stranger_rate_per_event = help_stranger_count / n_events if n_events else None
    help_stranger_rate_per_peer_event = (
        help_stranger_count / help_stranger_peer_events if help_stranger_peer_events else None
    )
    value_break_rate_per_event = value_break_count / n_events if n_events else None

    # Score trace arithmetic validity (v0): for utility actions where we emit contrib_sum/noise/score,
    # check that score ~= contrib_sum + noise (within tolerance).
    legible_n = 0
    legible_ok = 0
    for ev in events:
        factors = ev.explanation.factors
        if not isinstance(factors, dict):
            continue
        if "score" not in factors or "contrib_sum" not in factors or "noise" not in factors:
            continue
        try:
            score = float(factors["score"])
            contrib_sum = float(factors["contrib_sum"])
            noise = float(factors["noise"])
        except (TypeError, ValueError):
            continue
        legible_n += 1
        if abs(score - (contrib_sum + noise)) <= 1e-6:
            legible_ok += 1

    score_trace_arithmetic_validity = (legible_ok / legible_n) if legible_n else None

    # Belief decay analysis (v0): average per-tick change in belief confidence
    # for beliefs keyed by subject across consecutive ticks within an agent trace.
    conf_deltas: list[float] = []
    by_agent: dict[str, list[TraceEvent]] = {}
    for ev in events:
        by_agent.setdefault(ev.agent_id, []).append(ev)
    for aid, evs in by_agent.items():
        evs.sort(key=lambda e: e.t)
        prev = None
        for ev in evs:
            if prev is None:
                prev = ev
                continue
            prev_b = {b.subject: float(b.confidence) for b in prev.state_after.beliefs.values() if b.subject}
            cur_b = {b.subject: float(b.confidence) for b in ev.state_after.beliefs.values() if b.subject}
            for subj, c_prev in prev_b.items():
                if subj in cur_b:
                    conf_deltas.append(cur_b[subj] - c_prev)
            prev = ev
    belief_decay_avg_delta = (sum(conf_deltas) / len(conf_deltas)) if conf_deltas else None

    # Goal completion proxy rate (v0):
    # - For action-mapped goals: agent completes goal if it ever took the mapped action kind.
    # - For "survive": completed if final hunger >= 0.2 (very loose proxy).
    goal_proxy: dict[str, dict[str, bool]] = {}
    actions_by_agent: dict[str, set[str]] = {}
    for ev in events:
        actions_by_agent.setdefault(ev.agent_id, set()).add(ev.action.kind)

    for aid, last in final_by_agent.items():
        per_goal: dict[str, bool] = {}
        for gid in last.state_after.goals.keys():
            if gid == "survive":
                per_goal[gid] = float(last.state_after.needs["hunger"].value) >= 0.2
                continue
            ak = _goal_proxy_action_kind(gid)
            if ak is None:
                continue
            per_goal[gid] = ak in actions_by_agent.get(aid, set())
        if per_goal:
            goal_proxy[aid] = per_goal

    return {
        "ticks": int(max(ev.t for ev in events) + 1),
        "agents": sorted(final_by_agent.keys()),
        "avg_final_needs": avg_final_needs,
        "avg_trust_drift_abs": avg_trust_drift_abs,
        "avg_trust_drift_signed": avg_trust_drift_signed,
        "trust_drift_pair_count": trust_pairs,
        "unique_action_kinds": unique_kinds,
        "action_kind_counts": dict(counts),
        # Truth-labeled beliefs in final snapshot (denominator for Brier when labels exist).
        "belief_truth_labeled_count": belief_n,
        # Backward-compatible alias; prefer belief_truth_labeled_count in new tables/docs.
        "belief_count": belief_n,
        "belief_total_final": belief_total_final,
        "belief_truth_labeled_final": belief_n,
        "belief_truth_labeled_fraction": belief_truth_labeled_fraction,
        "belief_accuracy": belief_accuracy,
        "belief_brier": belief_brier,
        "belief_uptake": uptake_summary,
        "gossip_count": gossip_count,
        "gossip_rate_per_event": gossip_rate_per_event,
        "unique_belief_subjects_final": unique_belief_subjects_final,
        "belief_spread_agents": belief_spread_agents,
        "belief_spread_fraction": belief_spread_fraction,
        "primary_subject": primary_subject,
        "help_stranger_count": help_stranger_count,
        "help_stranger_peer_events": help_stranger_peer_events,
        "help_stranger_rate_per_event": help_stranger_rate_per_event,
        "help_stranger_rate_per_peer_event": help_stranger_rate_per_peer_event,
        "value_break_count": value_break_count,
        "value_break_rate_per_event": value_break_rate_per_event,
        "goal_completion_proxy": goal_proxy,
        "score_trace_arithmetic_validity": score_trace_arithmetic_validity,
        "belief_conf_delta_avg": belief_decay_avg_delta,
    }

