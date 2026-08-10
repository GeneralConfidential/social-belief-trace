from __future__ import annotations

import json
import tempfile
from pathlib import Path

from npc_agent_benchmark.harness.runner import run_episode


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    scenario_path = root / "scenarios" / "toy_provenance_stress_v0.json"
    scenario = json.loads(scenario_path.read_text(encoding="utf-8"))

    with tempfile.TemporaryDirectory() as tmp:
        trace_path = Path(tmp) / "trace.jsonl"
        summary = run_episode(
            scenario=scenario,
            ticks=70,
            seed=0,
            trace_path=trace_path,
            use_provenance_weighting=True,
            enable_rumor_channel=True,
            apply_belief_updates=True,
            policy_mode="utility_only",
        )
        if not trace_path.exists():
            raise SystemExit("smoke test did not write a trace")
        if summary.get("scenario_id") != "toy_provenance_stress_v0":
            raise SystemExit(f"unexpected scenario_id: {summary.get('scenario_id')!r}")

        for line_no, line in enumerate(trace_path.read_text(encoding="utf-8").splitlines(), 1):
            stripped = line.strip()
            if not stripped:
                continue
            event = json.loads(stripped)
            if event.get("trace_schema_version") != "trace_v0":
                raise SystemExit(
                    f"trace_schema_version missing/wrong on line {line_no} (expected trace_v0)"
                )

        uptake = summary.get("belief_uptake")
        if not isinstance(uptake, dict) or uptake.get("trust_conf_pearson_r_n") != 64:
            raise SystemExit("expected 64 source-sensitivity samples")

    print("Smoke test passed.")


if __name__ == "__main__":
    main()
