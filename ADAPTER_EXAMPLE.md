# External policy adapter example (v0)

This benchmark is intended to be **engine-agnostic**: an external NPC framework or policy can be evaluated as long as it can:

1. Accept a per-tick **observation** (at minimum: the agent’s needs and any delivered messages).
2. Emit an **action** (`kind`, optional `target`, optional `payload`).
3. Optionally emit an **explanation trace** (`ExplanationTrace`) — this is used only for artifact QA and future explanation studies.

The harness (`src/npc_agent_benchmark/harness/runner.py`) will then:

- apply toy world dynamics (need changes, social effects),
- deliver scheduled messages (rumors),
- broadcast witnessable actions,
- serialize a schema-versioned JSONL trace (`TraceEvent`, `trace_schema_version: "trace_v0"`).

## Where this plugs into the current code

1. **Policy interface**: Subclass `BaseAgent` in `src/npc_agent_benchmark/baselines/agents.py` and implement `act(self, *, t: int, state: AgentState, observation: dict[str, object]) -> tuple[Action, ExplanationTrace]` (same signature as the bundled baselines).
2. **Per-tick call site**: In `run_episode` (`harness/runner.py`), after constructing `obs` for each agent each tick, the harness invokes `action, expl = policies[aid].act(t=t, state=state, observation=obs)`. Any class stored in `policies[aid]` must satisfy that contract.
3. **Registration**: Register a `policy_mode` in `src/npc_agent_benchmark/harness/policy_registry.py` (`build_agent` / `REGISTERED_POLICY_MODES`). Entry-point plugins for third-party packages are left for later; this document pins the observation/action/explanation boundary.
4. **Trace schema**: Events are appended as JSON lines using `TraceEvent` in `src/npc_agent_benchmark/models.py`; lines include `"trace_schema_version": "trace_v0"` for artifact validation.

## What an adapter must implement

Conceptually, an adapter maps an external system’s state/action format onto:

- **Observation**:
  - `t`: tick index
  - `needs`: dict of `{"hunger","rest","safety","social"} -> float`
  - `messages`: list of delivered message dicts (may be empty)
  - `peer_ids`: list of other agent ids in the scenario
- **Action**:
  - `kind`: string action label (e.g., `"eat"`, `"rest"`, `"seek_safety"`, `"socialize"`, `"help_stranger"`, `"gossip"`, `"steal_food"`)
  - `target`: optional agent id (used by `help_stranger`/`gossip`)
  - `payload`: optional dict (used by `gossip` to carry `{"subject": ...}`)
- **ExplanationTrace** (optional):
  - `factors`: dict[str, float]
  - `notes`: list[str]

## Minimal adapter sketch

Below is a minimal, non-LLM adapter shape in pseudocode (not committed as code because external frameworks vary widely):

```python
class ExternalPolicyAdapter:
    def __init__(self, external_policy):
        self.external_policy = external_policy

    def act(self, *, t: int, state: AgentState, observation: dict[str, object]) -> tuple[Action, ExplanationTrace]:
        # 1) Map benchmark observation -> external policy inputs
        ext_obs = {
            "time": t,
            "needs": observation["needs"],
            "messages": observation["messages"],
            "peers": observation["peer_ids"],
        }

        # 2) Ask external policy for its decision (non-LLM)
        ext_action = self.external_policy.step(ext_obs)

        # 3) Map external action -> benchmark Action
        action = Action(kind=ext_action.kind, target=ext_action.target, payload=ext_action.payload or {})

        # 4) Optional explanation trace (can be empty)
        expl = ExplanationTrace(factors=getattr(ext_action, "factors", {}) or {}, notes=getattr(ext_action, "notes", []) or [])
        return action, expl
```

## What this does *not* claim

This document is a v0 trace/specification aid. It does not claim that any particular external framework cannot express these concepts; instead, it clarifies what is required to plug a new policy into the benchmark and emit comparable traces.

## Roadmap (not in v0)

- **Stable registration:** v0 now centralizes construction in `src/npc_agent_benchmark/harness/policy_registry.py` (`REGISTERED_POLICY_MODES`, `build_policies`). Entry-point / third-party package registration remains future work.
- **Minimal third-party demo:** `ExternalStubAgent` lives in `src/npc_agent_benchmark/baselines/external_stub.py` and is selectable via `policy_mode="external_stub"` (aggregated variant `external_stub_weighted_update`). `examples/external_stub_policy.py` is a thin import smoke check.

