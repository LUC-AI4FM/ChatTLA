#!/usr/bin/env python3
"""
curate_data.py — Quality-filter augmented training data to fix overfitting.

Problems identified:
  1. Bug-fix examples contain buggy TLA+ in user messages → model learns bad patterns
  2. Too many simple/repetitive specs (counters, toggles) swamp complex examples
  3. 334 augmented vs 57 expert examples (6:1 ratio) → drowns out quality data
  4. Some specs have structural issues (duplicate operators, missing TypeOK)

Strategy:
  - REMOVE all bug_fix examples (they pollute the model with buggy TLA+ patterns)
  - KEEP only spec_gen examples that are high quality
  - Quality filters: length, structure, no duplicate operators, SANY-valid
  - CAP at ~120 augmented examples (2:1 ratio vs expert data)
  - Prioritize complex/diverse specs over simple ones
"""

import json
import re
import subprocess
import tempfile
import os
import shutil
import sys
from pathlib import Path
from collections import Counter

_REPO_ROOT = Path(__file__).resolve().parents[1]
_AUGMENTED = _REPO_ROOT / "data" / "processed" / "augmented.jsonl"
_AUGMENTED_BACKUP = _REPO_ROOT / "data" / "processed" / "augmented.jsonl.backup"
_AUGMENTED_CURATED = _REPO_ROOT / "data" / "processed" / "augmented_curated.jsonl"
_JAR = _REPO_ROOT / "src" / "shared" / "tlc" / "tla2tools.jar"

# Quality criteria
MIN_SPEC_LINES = 20       # Filter trivial specs
MAX_AUGMENTED = 120        # Cap to maintain good ratio with expert data

def extract_spec(example: dict) -> str:
    """Extract the TLA+ spec from an augmented example."""
    for m in example.get("messages", []):
        if m.get("role") == "assistant" and "---- MODULE" in m.get("content", ""):
            return m["content"]
    return ""

def extract_user_prompt(example: dict) -> str:
    """Extract the user prompt."""
    for m in example.get("messages", []):
        if m.get("role") == "user":
            return m["content"]
    return ""

def is_bug_fix_example(example: dict) -> bool:
    """Check if this is a bug_fix example (contains buggy spec in user message)."""
    user = extract_user_prompt(example)
    return "syntax errors" in user.lower() or "sany errors" in user.lower() or "fix the spec" in user.lower()

def has_duplicate_operators(spec: str) -> bool:
    """Check if spec defines operators more than once (common model bug)."""
    defs = re.findall(r"^(\w+)\s*==", spec, re.MULTILINE)
    return len(defs) != len(set(defs))

def structural_score(spec: str) -> float:
    """Score spec structural completeness (0.0-1.0)."""
    checks = [
        bool(re.search(r"----\s*MODULE\s+\w+\s*----", spec)),  # MODULE header
        bool(re.search(r"EXTENDS\s+\w+", spec)),                # EXTENDS
        bool(re.search(r"VARIABLES?\s+", spec)),                 # VARIABLES
        bool(re.search(r"Init\s*==", spec)),                     # Init
        bool(re.search(r"Next\s*==", spec)),                     # Next
        bool(re.search(r"Spec\s*==", spec)),                     # Spec
        bool(re.search(r"TypeOK\s*==", spec)),                   # TypeOK
        bool(re.search(r"====", spec)),                          # End delimiter
    ]
    return sum(checks) / len(checks)

def complexity_score(spec: str) -> float:
    """Score spec complexity (0.0-1.0). Higher = more complex/interesting."""
    features = [
        bool(re.search(r"\\E\b|\\A\b", spec)),           # Quantifiers
        bool(re.search(r"SUBSET", spec)),                  # Set operations
        bool(re.search(r"\[.*->.*\]", spec)),              # Function types
        bool(re.search(r"Sequences", spec)),               # Sequences module
        bool(re.search(r"WF_|SF_", spec)),                 # Fairness
        bool(re.search(r"CONSTANT\s+\w+.*\n.*CONSTANT|CONSTANTS", spec)), # Multiple constants
        bool(re.search(r"RECURSIVE|INSTANCE|THEOREM", spec)),  # Advanced features
        spec.count("\n") >= 40,                            # Substantial length
        spec.count("\n") >= 60,                            # Large spec
        len(re.findall(r"^\w+\s*==", spec, re.MULTILINE)) >= 6,  # Many operators
    ]
    return sum(features) / len(features)

def sany_validate(spec: str) -> bool:
    """SANY-validate a spec. Returns True if valid."""
    m = re.search(r"MODULE\s+(\w+)", spec)
    mod_name = m.group(1) if m else "Spec"
    
    tmpdir = tempfile.mkdtemp()
    tla_path = os.path.join(tmpdir, f"{mod_name}.tla")
    with open(tla_path, "w") as f:
        f.write(spec)
    
    try:
        result = subprocess.run(
            ["java", "-cp", str(_JAR), "tla2sany.SANY", tla_path],
            capture_output=True, text=True, timeout=30
        )
        combined = result.stdout + result.stderr
        return not any(x in combined for x in ["Semantic errors", "Lexical error", "Parse Error", "Could not find module"])
    except Exception:
        return False
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

def topic_diversity_key(example: dict) -> str:
    """Extract topic category for diversity balancing."""
    user = extract_user_prompt(example).lower()
    topics = {
        "mutex": ["mutual exclusion", "mutex", "critical section"],
        "consensus": ["paxos", "raft", "consensus", "leader election", "vote"],
        "queue": ["queue", "producer", "consumer", "buffer", "fifo"],
        "lock": ["lock", "read-write", "semaphore"],
        "protocol": ["gossip", "token ring", "commit", "snapshot", "retransmission"],
        "algorithm": ["bakery", "peterson", "dekker", "dining", "philosopher"],
        "data_structure": ["stack", "key-value", "counter", "register", "dag"],
        "state_machine": ["traffic", "vending", "elevator", "door", "toggle", "clock"],
        "allocator": ["allocator", "resource", "broker", "publish"],
        "transaction": ["transaction", "isolation", "consistent"],
    }
    for category, keywords in topics.items():
        for kw in keywords:
            if kw in user:
                return category
    return "other"

def main():
    print("=" * 70)
    print("  ChatTLA Data Curation")
    print("=" * 70)
    
    # Load raw augmented data
    examples = []
    for line in _AUGMENTED.open():
        line = line.strip()
        if line:
            examples.append(json.loads(line))
    print(f"\nLoaded {len(examples)} augmented examples")
    
    # Phase 1: Remove bug-fix examples
    bug_fix = [e for e in examples if is_bug_fix_example(e)]
    spec_gen = [e for e in examples if not is_bug_fix_example(e)]
    print(f"  Bug-fix examples removed: {len(bug_fix)}")
    print(f"  Spec-gen examples kept:   {len(spec_gen)}")
    
    # Phase 2: Quality filters
    scored = []
    rejected_reasons = Counter()
    
    for ex in spec_gen:
        spec = extract_spec(ex)
        if not spec:
            rejected_reasons["no_spec"] += 1
            continue
        
        n_lines = spec.count("\n")
        if n_lines < MIN_SPEC_LINES:
            rejected_reasons["too_short"] += 1
            continue
        
        if has_duplicate_operators(spec):
            rejected_reasons["duplicate_ops"] += 1
            continue
        
        struct = structural_score(spec)
        if struct < 0.75:  # Must have at least 6/8 structural elements
            rejected_reasons["bad_structure"] += 1
            continue
        
        if not sany_validate(spec):
            rejected_reasons["sany_fail"] += 1
            continue
        
        # Compute scores for ranking
        cmplx = complexity_score(spec)
        topic = topic_diversity_key(ex)
        scored.append({
            "example": ex,
            "complexity": cmplx,
            "structural": struct,
            "topic": topic,
            "n_lines": n_lines,
        })
    
    print(f"\n  Quality filter results:")
    print(f"    Passed: {len(scored)}")
    for reason, count in rejected_reasons.most_common():
        print(f"    Rejected ({reason}): {count}")
    
    # Phase 3: Diversity-aware selection
    # Sort by complexity (highest first), then ensure topic diversity
    scored.sort(key=lambda x: (-x["complexity"], -x["n_lines"]))
    
    # Select with topic balancing: take up to MAX_PER_TOPIC from each topic
    selected = []
    topic_counts = Counter()
    MAX_PER_TOPIC = MAX_AUGMENTED // 8  # ~15 per topic
    
    # First pass: take from each topic up to limit
    for item in scored:
        topic = item["topic"]
        if topic_counts[topic] < MAX_PER_TOPIC:
            selected.append(item)
            topic_counts[topic] += 1
    
    # Second pass: fill remaining slots with highest complexity
    remaining = [s for s in scored if s not in selected]
    remaining.sort(key=lambda x: (-x["complexity"], -x["n_lines"]))
    for item in remaining:
        if len(selected) >= MAX_AUGMENTED:
            break
        selected.append(item)
    
    # Cap at MAX_AUGMENTED
    selected = selected[:MAX_AUGMENTED]
    
    print(f"\n  Selected {len(selected)} examples (max={MAX_AUGMENTED})")
    print(f"  Topic distribution:")
    final_topics = Counter(s["topic"] for s in selected)
    for topic, count in final_topics.most_common():
        print(f"    {topic:<20} {count}")
    
    print(f"\n  Complexity stats:")
    complexities = [s["complexity"] for s in selected]
    print(f"    Avg: {sum(complexities)/len(complexities):.2f}")
    print(f"    Min: {min(complexities):.2f}")
    print(f"    Max: {max(complexities):.2f}")
    
    line_counts = [s["n_lines"] for s in selected]
    print(f"  Line count stats:")
    print(f"    Avg: {sum(line_counts)/len(line_counts):.0f}")
    print(f"    <30: {sum(1 for l in line_counts if l < 30)}")
    print(f"    30-50: {sum(1 for l in line_counts if 30 <= l < 50)}")
    print(f"    50+: {sum(1 for l in line_counts if l >= 50)}")
    
    # Phase 4: Write curated data
    # Backup original
    if _AUGMENTED.exists():
        shutil.copy2(_AUGMENTED, _AUGMENTED_BACKUP)
        print(f"\n  Backed up original to {_AUGMENTED_BACKUP}")
    
    # Write curated
    with _AUGMENTED_CURATED.open("w") as f:
        for item in selected:
            f.write(json.dumps(item["example"], ensure_ascii=False) + "\n")
    print(f"  Wrote curated data to {_AUGMENTED_CURATED}")
    
    # Replace original with curated
    shutil.copy2(_AUGMENTED_CURATED, _AUGMENTED)
    print(f"  Replaced {_AUGMENTED} with curated data ({len(selected)} examples)")
    
    print(f"\n{'=' * 70}")
    print(f"  Data curation complete: {len(examples)} → {len(selected)} examples")
    print(f"  Bug-fix removed: {len(bug_fix)}")
    print(f"  Quality filtered: {len(spec_gen) - len(scored)}")
    print(f"  Diversity capped: {max(0, len(scored) - len(selected))}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
