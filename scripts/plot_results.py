from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt


def _mean(xs: list[float]) -> float | None:
    return (sum(xs) / len(xs)) if xs else None


def _std(xs: list[float]) -> float | None:
    if len(xs) < 2:
        return None
    m = sum(xs) / len(xs)
    v = sum((x - m) ** 2 for x in xs) / (len(xs) - 1)
    return math.sqrt(v)


def _load_rows(path: Path) -> list[dict[str, object]]:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    runs_dir = root / "runs"
    raw_path = runs_dir / "summary_raw.json"
    if not raw_path.exists():
        raise SystemExit("Missing runs/summary_raw.json. Run: uv run python scripts/run_all_agg.py")

    rows = _load_rows(raw_path)
    fig_dir = runs_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    # Plot: belief calibration (Brier) and source sensitivity on a high-sample provenance scenario.
    scenario_id = "toy_provenance_stress_v0"
    variants = [
        "routine_weighted_update",
        "utility_no_belief_update",
        "utility_unweighted_belief_update",
        "utility_weighted_belief_update",
        "utility_no_communication",
        "reputation_probe_weighted_update",
        "external_stub_weighted_update",
        "reference_weighted_action_use",
    ]
    variant_labels = {
        "routine_weighted_update": "Routine\nweighted",
        "utility_no_belief_update": "Utility\nno beliefs",
        "utility_unweighted_belief_update": "Utility\nunweighted",
        "utility_weighted_belief_update": "Utility\nweighted",
        "utility_no_communication": "Utility\nno comm",
        "reputation_probe_weighted_update": "Reputation\nprobe",
        "external_stub_weighted_update": "External\nstub",
        "reference_weighted_action_use": "Reference\nuses beliefs",
    }

    brier_by_variant: dict[str, list[float]] = defaultdict(list)
    r_by_variant: dict[str, list[float]] = defaultdict(list)
    r_n_by_variant: dict[str, list[float]] = defaultdict(list)

    for r in rows:
        if str(r.get("scenario_id")) != scenario_id:
            continue
        vid = str(r.get("variant_id"))
        if vid not in variants:
            continue
        bb = r.get("belief_brier")
        if isinstance(bb, (int, float)):
            brier_by_variant[vid].append(float(bb))
        bu = r.get("belief_uptake")
        if isinstance(bu, dict):
            rr = bu.get("trust_conf_pearson_r")
            if isinstance(rr, (int, float)):
                r_by_variant[vid].append(float(rr))
            rn = bu.get("trust_conf_pearson_r_n")
            if isinstance(rn, (int, float)):
                r_n_by_variant[vid].append(float(rn))

    # Figure 1: Brier score mean±std
    xs = list(range(len(variants)))
    means = [_mean(brier_by_variant[v]) for v in variants]
    stds = [_std(brier_by_variant[v]) for v in variants]

    labels = [variant_labels[v] for v in variants]

    plt.figure(figsize=(10.2, 4.0))
    plt.bar(xs, [m or 0.0 for m in means], yerr=[s or 0.0 for s in stds], capsize=4)
    plt.xticks(xs, labels, rotation=0, ha="center")
    plt.title("Provenance stress: retained-belief Brier (not a policy ranking)")
    plt.ylabel("Brier score over retained final beliefs\n(lower is better; coverage differs by variant)")
    for i, m in enumerate(means):
        if m is None:
            plt.text(i, 0.01, "no retained\nbeliefs", ha="center", va="bottom", fontsize=8)
    plt.tight_layout()
    out1 = fig_dir / "fig1_brier_provenance_stress.png"
    plt.savefig(out1, dpi=200)
    plt.close()

    # Figure 2: source sensitivity r mean±std (when available).
    # Note: r is suppressed upstream when n_messages < 5, so some variants may have no samples.
    means_r = [_mean(r_by_variant[v]) for v in variants]
    stds_r = [_std(r_by_variant[v]) for v in variants]
    means_rn = [_mean(r_n_by_variant[v]) for v in variants]

    plt.figure(figsize=(10.2, 4.0))
    # Use NaN for missing r so we don't visually imply "0 correlation".
    heights = [m if m is not None else math.nan for m in means_r]
    yerrs = [s if s is not None else 0.0 for s in stds_r]
    plt.bar(xs, heights, yerr=yerrs, capsize=4)
    plt.axhline(0.0, color="black", linewidth=0.8)
    plt.xticks(xs, labels, rotation=0, ha="center")
    plt.ylim(-1.05, 1.05)
    plt.ylabel("Source sensitivity\nPearson r(trust, accepted confidence)")
    plt.title("Provenance stress: trust-to-confidence sensitivity (harness check)")
    # Annotate missing samples (suppressed r) with the underlying message count.
    for i, (m, rn) in enumerate(zip(means_r, means_rn, strict=True)):
        if m is None:
            n_txt = f"n={int(rn) if rn is not None else 0}"
            plt.text(i, 0.02, f"(suppressed)\n{n_txt}", ha="center", va="bottom", fontsize=7)
    plt.tight_layout()
    out2 = fig_dir / "fig2_source_sensitivity_provenance_stress.png"
    plt.savefig(out2, dpi=200)
    plt.close()

    print(f"Wrote: {out1}")
    print(f"Wrote: {out2}")


if __name__ == "__main__":
    main()

