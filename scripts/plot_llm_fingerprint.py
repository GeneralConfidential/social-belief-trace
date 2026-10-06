"""Plot Brier vs gossip fingerprint for RQ6 LLM vs symbolic heads."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt


def _mean(xs: list[float]) -> float | None:
    return (sum(xs) / len(xs)) if xs else None


def _load_means(path: Path, *, backend_label: str) -> list[dict[str, object]]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    buckets: dict[tuple[str, str], dict[str, list[float]]] = defaultdict(
        lambda: {"gossip_count": [], "belief_brier": []}
    )
    for r in rows:
        pol = str(r.get("policy_mode"))
        sid = str(r.get("scenario_id"))
        if pol not in {"llm_openai", "llm_local", "utility_only", "reference"}:
            continue
        if sid not in {"toy_rumor_v0", "pubhealth_conflicting_guidance_v0", "toy_provenance_stress_v0"}:
            continue
        key = (pol, sid)
        g = r.get("gossip_count")
        b = r.get("belief_brier")
        if isinstance(g, (int, float)):
            buckets[key]["gossip_count"].append(float(g))
        if isinstance(b, (int, float)):
            buckets[key]["belief_brier"].append(float(b))

    out: list[dict[str, object]] = []
    for (pol, sid), vals in sorted(buckets.items()):
        out.append(
            {
                "backend_label": backend_label,
                "policy_mode": pol,
                "scenario_id": sid,
                "gossip_mean": _mean(vals["gossip_count"]),
                "brier_mean": _mean(vals["belief_brier"]),
            }
        )
    return out


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    points = []
    points.extend(_load_means(root / "runs" / "llm_eval" / "summary_raw_openai.json", backend_label="openai"))
    # Prefer openai symbolic+llm for the primary scatter; add local LLM rumor/pubhealth only.
    ollama = _load_means(root / "runs" / "llm_eval" / "summary_raw.json", backend_label="ollama")
    points.extend([p for p in ollama if p["policy_mode"] == "llm_local"])

    fig_dir = root / "runs" / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    markers = {
        "toy_rumor_v0": "o",
        "pubhealth_conflicting_guidance_v0": "s",
        "toy_provenance_stress_v0": "^",
    }
    colors = {
        "llm_openai": "#1f77b4",
        "llm_local": "#ff7f0e",
        "utility_only": "#2ca02c",
        "reference": "#9467bd",
    }
    labels_done: set[str] = set()

    fig, ax = plt.subplots(figsize=(5.2, 3.8))
    for p in points:
        g = p["gossip_mean"]
        b = p["brier_mean"]
        if not isinstance(g, float) or not isinstance(b, float):
            continue
        pol = str(p["policy_mode"])
        sid = str(p["scenario_id"])
        legend = f"{pol}/{sid.split('_')[1] if '_' in sid else sid}"
        # Shorter legend keys
        short = {
            ("llm_openai", "toy_rumor_v0"): "gpt-4o-mini / rumor",
            ("llm_openai", "pubhealth_conflicting_guidance_v0"): "gpt-4o-mini / pubhealth",
            ("llm_openai", "toy_provenance_stress_v0"): "gpt-4o-mini / stress",
            ("llm_local", "toy_rumor_v0"): "ollama-3B / rumor",
            ("llm_local", "pubhealth_conflicting_guidance_v0"): "ollama-3B / pubhealth",
            ("llm_local", "toy_provenance_stress_v0"): "ollama-3B / stress",
            ("utility_only", "toy_rumor_v0"): "utility / rumor",
            ("utility_only", "pubhealth_conflicting_guidance_v0"): "utility / pubhealth",
            ("utility_only", "toy_provenance_stress_v0"): "utility / stress",
            ("reference", "toy_rumor_v0"): "reference / rumor",
            ("reference", "pubhealth_conflicting_guidance_v0"): "reference / pubhealth",
            ("reference", "toy_provenance_stress_v0"): "reference / stress",
        }.get((pol, sid), f"{pol}/{sid}")
        lab = short if short not in labels_done else None
        if lab:
            labels_done.add(short)
        ax.scatter(
            g,
            b,
            c=colors.get(pol, "gray"),
            marker=markers.get(sid, "o"),
            s=55,
            label=lab,
            edgecolors="black",
            linewidths=0.4,
            zorder=3,
        )

    ax.set_xlabel("Mean gossip count")
    ax.set_ylabel("Mean Brier (truth-labeled beliefs)")
    ax.set_title("RQ6 fingerprint: Brier vs gossip")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7, loc="best", framealpha=0.9)
    fig.tight_layout()
    out = fig_dir / "fig3_brier_vs_gossip_rq6.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
