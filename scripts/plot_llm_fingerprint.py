"""Plot Brier vs gossip fingerprint for RQ6 LLM vs symbolic heads.

Three panels so provenance-stress gossip is not mixed into a fair ranking.
"""

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
    ollama = _load_means(root / "runs" / "llm_eval" / "summary_raw.json", backend_label="ollama")
    points.extend([p for p in ollama if p["policy_mode"] == "llm_local"])

    fig_dir = root / "runs" / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    colors = {
        "llm_openai": "#1f77b4",
        "llm_local": "#ff7f0e",
        "utility_only": "#2ca02c",
        "reference": "#9467bd",
    }
    short_pol = {
        "llm_openai": "gpt-4o-mini",
        "llm_local": "ollama-3B",
        "utility_only": "utility",
        "reference": "reference",
    }
    panels = [
        ("toy_rumor_v0", "Rumor (shared gossip targets)", False),
        ("pubhealth_conflicting_guidance_v0", "Pubhealth (shared gossip targets)", False),
        ("toy_provenance_stress_v0", "Provenance stress (unequal targets)", True),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(11.0, 3.6), sharey=True)
    legend_done: set[str] = set()
    for ax, (sid, title, unequal) in zip(axes, panels, strict=True):
        for p in points:
            if p["scenario_id"] != sid:
                continue
            g = p["gossip_mean"]
            b = p["brier_mean"]
            if not isinstance(g, float) or not isinstance(b, float):
                continue
            pol = str(p["policy_mode"])
            lab = short_pol.get(pol, pol)
            use_lab = lab if lab not in legend_done else None
            if use_lab:
                legend_done.add(lab)
            ax.scatter(
                g,
                b,
                c=colors.get(pol, "gray"),
                s=55,
                label=use_lab,
                edgecolors="black",
                linewidths=0.4,
                zorder=3,
            )
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("Mean gossip count", fontsize=10)
        ax.tick_params(axis="both", labelsize=9)
        ax.grid(True, alpha=0.3)
        if unequal:
            ax.set_facecolor("#f6f6f6")
    axes[0].set_ylabel("Mean Brier (retained labeled beliefs)", fontsize=10)
    handles, labels = [], []
    for ax in axes:
        h, lab = ax.get_legend_handles_labels()
        handles.extend(h)
        labels.extend(lab)
    # Deduplicate legend labels while preserving order.
    seen: set[str] = set()
    uniq_h, uniq_l = [], []
    for h, lab in zip(handles, labels, strict=True):
        if lab in seen:
            continue
        seen.add(lab)
        uniq_h.append(h)
        uniq_l.append(lab)
    fig.legend(uniq_h, uniq_l, loc="upper center", ncol=4, fontsize=9, framealpha=0.95)
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    out = fig_dir / "fig3_brier_vs_gossip_rq6.png"
    fig.savefig(out, dpi=300)
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
