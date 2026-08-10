"""Minimal third-party-style policy stub for adapter demonstrations.

The class now lives in the package as ``npc_agent_benchmark.baselines.external_stub.ExternalStubAgent``.
It is registered under ``policy_mode="external_stub"`` (see ``VARIANTS`` in ``scripts/run_all_agg.py``).

Smoke-import from ``beliefbench/``::

    uv run python examples/external_stub_policy.py
"""

from __future__ import annotations

from npc_agent_benchmark.baselines.external_stub import ExternalStubAgent


if __name__ == "__main__":
    print("external_stub_policy:", ExternalStubAgent.__name__, "import OK")
