"""RQ6 provenance-stress autopsy from existing llm_eval traces (no API)."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path


def _hist(path: Path) -> dict[str, object]:
    kinds: Counter[str] = Counter()
    peer_lens: Counter[int] = Counter()
    examples: list[dict[str, object]] = []
    fallback_notes = 0
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        ev = json.loads(line)
        kind = str(ev.get("action", {}).get("kind", "?"))
        kinds[kind] += 1
        peers = ev.get("observation", {}).get("peer_ids") or []
        peer_lens[len(peers) if isinstance(peers, list) else -1] += 1
        notes = ev.get("explanation", {}).get("notes") or []
        if any("fallback" in str(n).lower() or "failure" in str(n).lower() for n in notes):
            fallback_notes += 1
        if len(examples) < 3:
            examples.append(
                {
                    "t": ev.get("t"),
                    "action": ev.get("action"),
                    "peer_ids": peers,
                    "notes": notes[:3],
                }
            )
    return {
        "trace": path.as_posix(),
        "action_kind_counts": dict(kinds),
        "peer_ids_len_counts": {str(k): v for k, v in sorted(peer_lens.items())},
        "fallbackish_note_events": fallback_notes,
        "examples": examples,
    }


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    runs = root / "runs" / "llm_eval"
    out_dir = runs / "autopsy"
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for policy in ("llm_openai", "llm_local", "utility_only", "reference"):
        path = runs / f"toy_provenance_stress_v0__{policy}__seed0.jsonl"
        if path.exists():
            rows.append({"policy_mode": policy, "seed": 0, **_hist(path)})

    finding = {
        "scenario_id": "toy_provenance_stress_v0",
        "finding": (
            "Single simulated recipient (agent a); observation.peer_ids is always empty. "
            "Constrained LLM gossip requires a peer_id target, so gossip is structurally "
            "impossible. Symbolic utility/reference policies still emit gossip by targeting "
            "relationship/source ids (message senders), which are not in peer_ids. "
            "OpenAI gpt-4o-mini: 0% invalid, 100% eat (schema-OK, non-gossip). "
            "Ollama local: high rest/fallback rate under the same empty peer_ids constraint."
        ),
        "rows": rows,
    }

    json_path = out_dir / "provenance_stress_autopsy.json"
    json_path.write_text(json.dumps(finding, indent=2, sort_keys=True), encoding="utf-8")

    md_lines = [
        "# RQ6 provenance-stress autopsy",
        "",
        finding["finding"],
        "",
    ]
    for row in rows:
        md_lines.append(f"## {row['policy_mode']} (seed 0)")
        md_lines.append(f"- action_kind_counts: `{row['action_kind_counts']}`")
        md_lines.append(f"- peer_ids_len_counts: `{row['peer_ids_len_counts']}`")
        md_lines.append(f"- fallbackish_note_events: {row['fallbackish_note_events']}")
        md_lines.append("- examples:")
        for ex in row["examples"]:
            md_lines.append(f"  - t={ex['t']} action={ex['action']} peer_ids={ex['peer_ids']}")
        md_lines.append("")
    md_path = out_dir / "provenance_stress_autopsy.md"
    md_path.write_text("\n".join(md_lines), encoding="utf-8")
    print(md_path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
