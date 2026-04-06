#!/usr/bin/env python3
"""Merge all diamond judgment files into the canonical judgments JSONL."""
import json
import glob
import logging
from pathlib import Path

log = logging.getLogger("merge_judgments")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

_REPO_ROOT = Path(__file__).resolve().parents[1]
_LOGS_DIR = _REPO_ROOT / "outputs" / "logs"
_CANONICAL = _LOGS_DIR / "diamond_judgments.jsonl"


def merge():
    # Find all judgment files
    patterns = [
        str(_LOGS_DIR / "diamond_judgments*.jsonl"),
    ]
    all_files = set()
    for pat in patterns:
        all_files.update(glob.glob(pat))

    seen: dict[str, dict] = {}
    for fpath in sorted(all_files):
        count = 0
        with open(fpath) as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    obj = json.loads(line)
                    pid = obj.get("prompt_id", "")
                    if pid and pid not in seen:
                        seen[pid] = obj
                        count += 1
                except json.JSONDecodeError:
                    continue
        log.info(f"  {Path(fpath).name}: {count} new judgments")

    # Write canonical file
    with open(_CANONICAL, "w") as f:
        for obj in seen.values():
            f.write(json.dumps(obj) + "\n")

    log.info(f"Merged {len(seen)} unique judgments -> {_CANONICAL}")

    # Score summary
    scores = [j.get("avg_score", 0) for j in seen.values()]
    if scores:
        verdicts = {}
        for j in seen.values():
            v = j.get("verdict", "?")
            verdicts[v] = verdicts.get(v, 0) + 1
        log.info(f"Scores: min={min(scores):.1f} max={max(scores):.1f} "
                 f"mean={sum(scores)/len(scores):.1f} "
                 f"median={sorted(scores)[len(scores)//2]:.1f}")
        log.info(f"Verdicts: {verdicts}")

    return seen


if __name__ == "__main__":
    merge()
