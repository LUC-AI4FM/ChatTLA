#!/usr/bin/env python3
"""Merge reasoning files and inject into diamond_curated.jsonl."""
import json
import glob
import logging
from pathlib import Path

log = logging.getLogger("merge_reasoning")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

_REPO_ROOT = Path(__file__).resolve().parents[1]
_LOGS_DIR = _REPO_ROOT / "outputs" / "logs"
_CURATED = _REPO_ROOT / "data" / "processed" / "diamond_curated.jsonl"
_JUDGMENTS = _LOGS_DIR / "diamond_judgments.jsonl"


def merge():
    # Load all reasoning files
    reasoning_map: dict[str, str] = {}
    for fpath in sorted(glob.glob(str(_LOGS_DIR / "diamond_reasoning*.jsonl"))):
        count = 0
        with open(fpath) as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    obj = json.loads(line)
                    pid = obj.get("prompt_id", "")
                    cot = obj.get("chain_of_thought", "")
                    if pid and cot and pid not in reasoning_map:
                        reasoning_map[pid] = cot
                        count += 1
                except json.JSONDecodeError:
                    continue
        log.info(f"  {Path(fpath).name}: {count} new reasoning entries")

    log.info(f"Total reasoning: {len(reasoning_map)}")

    # Also inject reasoning into judgments file for the pipeline
    if _JUDGMENTS.exists():
        judgments = []
        with open(_JUDGMENTS) as f:
            for line in f:
                if line.strip():
                    try:
                        obj = json.loads(line)
                        pid = obj.get("prompt_id", "")
                        if pid in reasoning_map:
                            obj["chain_of_thought"] = reasoning_map[pid]
                        judgments.append(obj)
                    except json.JSONDecodeError:
                        pass
        with open(_JUDGMENTS, "w") as f:
            for obj in judgments:
                f.write(json.dumps(obj) + "\n")
        enriched = sum(1 for j in judgments if j.get("chain_of_thought"))
        log.info(f"Enriched {enriched} judgments with reasoning")

    # Inject into curated dataset
    if _CURATED.exists():
        records = []
        injected = 0
        with open(_CURATED) as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    obj = json.loads(line)
                    pid = obj.get("_prompt_id", "")
                    cot = reasoning_map.get(pid, "")
                    if cot:
                        # Update the analysis channel message
                        msgs = obj.get("messages", [])
                        for m in msgs:
                            if m.get("channel") == "analysis":
                                m["content"] = cot
                                injected += 1
                                break
                    records.append(obj)
                except json.JSONDecodeError:
                    pass

        with open(_CURATED, "w") as f:
            for obj in records:
                f.write(json.dumps(obj) + "\n")
        log.info(f"Injected reasoning into {injected}/{len(records)} curated specs")

    return reasoning_map


if __name__ == "__main__":
    merge()
