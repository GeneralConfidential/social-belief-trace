"""Write ``schemas/trace_v0.schema.json`` from the Pydantic ``TraceEvent`` model."""

from __future__ import annotations

import json
from pathlib import Path


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    out_dir = root / "schemas"
    out_dir.mkdir(parents=True, exist_ok=True)
    from npc_agent_benchmark.models import TraceEvent

    schema = TraceEvent.model_json_schema()
    path = out_dir / "trace_v0.schema.json"
    path.write_text(json.dumps(schema, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
