from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


NeedName = Literal["hunger", "rest", "safety", "social"]


class NeedState(BaseModel):
    value: float = Field(ge=0.0, le=1.0)


class Relationship(BaseModel):
    trust: float = Field(ge=-1.0, le=1.0, default=0.0)


class Belief(BaseModel):
    subject: str = Field(default="")
    proposition: str
    source: str | None = None
    confidence: float = Field(ge=0.0, le=1.0, default=0.5)
    truth: bool | None = None  # for benchmark evaluation only


class Goal(BaseModel):
    description: str
    priority: float = Field(ge=0.0, le=1.0, default=0.5)


class Value(BaseModel):
    description: str
    rigidity: float = Field(ge=0.0, le=1.0, default=0.5)


class AgentState(BaseModel):
    agent_id: str
    needs: dict[NeedName, NeedState]
    relationships: dict[str, Relationship] = Field(default_factory=dict)
    beliefs: dict[str, Belief] = Field(default_factory=dict)  # key = belief_id
    goals: dict[str, Goal] = Field(default_factory=dict)
    values: dict[str, Value] = Field(default_factory=dict)


class Action(BaseModel):
    kind: str
    target: str | None = None
    payload: dict[str, object] = Field(default_factory=dict)


class ExplanationTrace(BaseModel):
    factors: dict[str, float] = Field(default_factory=dict)
    notes: list[str] = Field(default_factory=list)


class TraceEvent(BaseModel):
    trace_schema_version: str = "trace_v0"
    t: int
    agent_id: str
    observation: dict[str, object]
    action: Action
    explanation: ExplanationTrace
    state_after: AgentState

