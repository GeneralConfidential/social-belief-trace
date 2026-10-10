"""One-off check: committed llm_eval summaries vs manuscript table constants."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Manuscript Table VI (tab:llm-rq6) — 5-seed means
TABLE_VI: dict[tuple[str, str], dict[str, float]] = {
    ("llm_openai", "toy_rumor_v0"): {"gossip_count": 9.0, "belief_brier": 0.11},
    ("utility_only", "toy_rumor_v0"): {"gossip_count": 18.8, "belief_brier": 0.13},
}

# Manuscript Table VII (tab:multimodel) — 3-seed means (subset)
TABLE_VII: list[tuple[str, str, str, float | None, float | None, float | None]] = [
    ("gpt-4o-mini", "toy_rumor_v0", "gossip_count", 9.0, 0.0, 1.0),
    ("gpt-4o", "toy_rumor_v0", "gossip_count", 0.0, 0.0, 1.0),
    ("gpt-6.1-sol", "toy_rumor_v0", "gossip_count", 0.0, 0.01, 1.0),
    ("llama3.2-3b", "toy_rumor_v0", "gossip_count", 30.0, 0.0, 1.0),
    ("gpt-4o", "toy_adversarial_injection_v0", "gossip_count", 0.0, 0.0, 1.0),
    ("gpt-6.1-sol", "toy_adversarial_injection_v0", "gossip_count", 0.0, 0.0, 1.0),
]


def _mean(rows: list[dict], *, policy: str, scenario: str, key: str) -> float | None:
    xs: list[float] = []
    for row in rows:
        if row.get("policy_mode") != policy or row.get("scenario_id") != scenario:
            continue
        val = row.get(key)
        if isinstance(val, (int, float)):
            xs.append(float(val))
    return (sum(xs) / len(xs)) if xs else None


def _close(a: float | None, b: float, tol: float = 0.05) -> bool:
    if a is None:
        return False
    return abs(a - b) <= tol


def main() -> int:
    failures: list[str] = []

    raw_openai = json.loads((ROOT / "runs/llm_eval/summary_raw_openai.json").read_text(encoding="utf-8"))
    for (policy, scenario), expected in TABLE_VI.items():
        for key, paper_val in expected.items():
            got = _mean(raw_openai, policy=policy, scenario=scenario, key=key)
            if not _close(got, paper_val, tol=0.15 if key == "belief_brier" else 0.05):
                failures.append(f"Table VI {policy}/{scenario} {key}: paper={paper_val} json={got}")

    for sub in ("multimodel", "multimodel_h", "multimodel_sol"):
        agg_path = ROOT / "runs/llm_eval" / sub / "summary_agg.json"
        agg = json.loads(agg_path.read_text(encoding="utf-8"))
        by_key = {(str(r["model_label"]), str(r["scenario_id"])): r for r in agg}
        for model, scenario, _metric, g_p, inv_p, acc_p in TABLE_VII:
            if sub == "multimodel" and model in ("gpt-4o", "gpt-6.1-sol", "qwen2.5-7b"):
                continue
            if sub == "multimodel_h" and model not in ("gpt-4o", "qwen2.5-7b"):
                continue
            if sub == "multimodel_sol" and model != "gpt-6.1-sol":
                continue
            row = by_key.get((model, scenario))
            if row is None:
                continue
            g = row.get("gossip_count_mean")
            inv = row.get("llm_invalid_action_rate_mean")
            acc = row.get("belief_accuracy_mean")
            if isinstance(g, (int, float)) and not _close(float(g), g_p, 0.2):
                failures.append(f"Table VII {model}/{scenario} gossip: paper={g_p} json={g}")
            if inv_p is not None and isinstance(inv, (int, float)) and not _close(float(inv), inv_p, 0.02):
                failures.append(f"Table VII {model}/{scenario} inv: paper={inv_p} json={inv}")
            if acc_p is not None and isinstance(acc, (int, float)) and not _close(float(acc), acc_p, 0.02):
                failures.append(f"Table VII {model}/{scenario} acc: paper={acc_p} json={acc}")

    sidecar = json.loads(
        (ROOT / "runs/llm_eval/multimodel_sol/toy_rumor_v0__gpt-6.1-sol__seed0.metrics.json").read_text(
            encoding="utf-8"
        )
    )
    for field in ("model_label", "model_id"):
        if sidecar.get(field) != "gpt-6.1-sol":
            failures.append(f"gpt-6.1-sol sidecar {field}={sidecar.get(field)!r}")

    if failures:
        print("VERIFY FAILED:")
        for line in failures:
            print(" ", line)
        return 1
    print("VERIFY OK: Table VI/VII spot-checks match committed llm_eval summaries; gpt-6.1-sol id confirmed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
