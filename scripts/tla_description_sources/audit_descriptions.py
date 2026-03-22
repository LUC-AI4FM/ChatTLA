#!/usr/bin/env python3
"""Validate harvested descriptions JSON (exit 0 if all rows resolved with non-empty text)."""

import json
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
_DEFAULT = _REPO / "data" / "derived" / "tla_descriptions.json"


def main() -> int:
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else _DEFAULT
    data = json.loads(path.read_text(encoding="utf-8"))
    issues = []
    by_repo = {"tlaplus/Examples": 0, "tlaplus/tlapm": 0}
    for row in data:
        spec = row.get("module_name") or row.get("specification") or "?"
        rid = row.get("id")
        if row.get("confidence") == "none" or "not_found" in row.get("provenance", []):
            issues.append(f"id={rid} module={spec}: not resolved")
        desc = row.get("description")
        if isinstance(desc, dict):
            nar = (desc.get("narrative") or "").strip()
            if not nar:
                issues.append(f"id={rid} module={spec}: empty description.narrative")
        elif isinstance(desc, str):
            if not desc.strip():
                issues.append(f"id={rid} module={spec}: empty description")
        else:
            issues.append(f"id={rid} module={spec}: missing or invalid description")
        sr = row.get("source_repository", "")
        if sr in by_repo:
            by_repo[sr] += 1
    print(f"Rows: {len(data)}")
    print(f"By source: {by_repo}")
    if issues:
        print("ISSUES:")
        for i in issues:
            print(" ", i)
        return 1
    print("OK — all descriptions non-empty and resolved.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
