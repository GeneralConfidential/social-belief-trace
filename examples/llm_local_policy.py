"""Example entry point for the local LLM policy (Ollama).

Smoke-import from package root::

    uv run python examples/llm_local_policy.py

Full run requires Ollama (see LLM_POLICY.md)::

    uv run npc-agent-benchmark --scenario scenarios/toy_scarcity_v0.json \\
        --out runs/llm_local_smoke.jsonl --ticks 5 --policy-mode llm_local
"""

from __future__ import annotations

from npc_agent_benchmark.baselines.llm_local import ConstrainedLlmAgent


if __name__ == "__main__":
    print("llm_local_policy:", ConstrainedLlmAgent.__name__, "import OK")
