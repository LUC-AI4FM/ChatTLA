#!/usr/bin/env python3
"""Batch 4 of hand-crafted Diamond TLA+ specs."""
from __future__ import annotations
import json, logging, sys
from datetime import datetime
from pathlib import Path

log = logging.getLogger("gen_diamond_b4")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

_REPO_ROOT = Path(__file__).resolve().parents[1]
_DIAMOND_OUT = _REPO_ROOT / "data" / "processed" / "diamond_sft.jsonl"

SPECS: list[tuple[str, str, str]] = []

# ---------------------------------------------------------------------------
# 1. Inventory management
# ---------------------------------------------------------------------------
SPECS.append(("manual4_inventory", "Model an inventory management system with restock and sell actions. Stock is bounded by a maximum capacity. Track quantity on hand and ensure it never goes negative or exceeds capacity.", r"""---- MODULE InventoryMgmt ----
EXTENDS Integers

CONSTANT Max

VARIABLE stock

vars == <<stock>>

Init == stock = 0

Restock == stock < Max /\ stock' = stock + 1

Sell == stock > 0 /\ stock' = stock - 1

Next == Restock \/ Sell \/ UNCHANGED vars

Spec == Init /\ [][Next]_vars

TypeOK == stock \in 0..Max

SafetyBounded == stock >= 0 /\ stock <= Max
====
"""))

# ---------------------------------------------------------------------------
# 2. Chat room
# ---------------------------------------------------------------------------
SPECS.append(("manual4_chatroom", "Model a chat room where participants can join or leave. The number of participants is bounded by a maximum capacity. Track the count of active participants.", r"""---- MODULE ChatRoom ----
EXTENDS Integers

CONSTANT Max

VARIABLE members

vars == <<members>>

Init == members = 0

Join == members < Max /\ members' = members + 1

Leave == members > 0 /\ members' = members - 1

Next == Join \/ Leave \/ UNCHANGED vars

Spec == Init /\ [][Next]_vars

TypeOK == members \in 0..Max

SafetyBounded == members >= 0 /\ members <= Max
====
"""))

# ---------------------------------------------------------------------------
# 3. Elevator
# ---------------------------------------------------------------------------
SPECS.append(("manual4_elevator", "Model an elevator system that moves between floors, with a direction indicator. The elevator can move up, move down, or stay idle. Floors are bounded.", r"""---- MODULE ElevatorCtrl ----
EXTENDS Integers

CONSTANT Max

VARIABLES floor, dir

vars == <<floor, dir>>

Init == floor = 1 /\ dir = 0

MoveUp == floor < Max /\ floor' = floor + 1 /\ dir' = 1

MoveDown == floor > 1 /\ floor' = floor - 1 /\ dir' = -1

Idle == floor' = floor /\ dir' = 0

Next == MoveUp \/ MoveDown \/ Idle \/ UNCHANGED vars

Spec == Init /\ [][Next]_vars

TypeOK == floor \in 1..Max /\ dir \in {-1, 0, 1}

SafetyBounded == floor >= 1 /\ floor <= Max
====
"""))

# ---------------------------------------------------------------------------
# 4. Version control (simplified)
# ---------------------------------------------------------------------------
SPECS.append(("manual4_vcs", "Model a simplified version control system tracking the number of commits and branches. Users can commit (incrementing the commit counter) or create a branch. Both are bounded.", r"""---- MODULE VersionCtrl ----
EXTENDS Integers

CONSTANT Max

VARIABLES commits, branches

vars == <<commits, branches>>

Init == commits = 0 /\ branches = 1

Commit == commits < Max /\ commits' = commits + 1 /\ UNCHANGED branches

Branch == branches < Max /\ branches' = branches + 1 /\ UNCHANGED commits

Next == Commit \/ Branch \/ UNCHANGED vars

Spec == Init /\ [][Next]_vars

TypeOK == commits \in 0..Max /\ branches \in 1..Max

SafetyBounded == commits >= 0 /\ commits <= Max /\ branches >= 1 /\ branches <= Max
====
"""))

# ---------------------------------------------------------------------------
# 5. Task dependency DAG execution
# ---------------------------------------------------------------------------
SPECS.append(("manual4_taskdag", "Model a task execution system with three tasks where task B depends on task A and task C depends on task B. Tasks transition from pending to running to done.", r"""---- MODULE TaskDag ----
EXTENDS Integers

VARIABLES a, b, c

vars == <<a, b, c>>

Init == a = 0 /\ b = 0 /\ c = 0

RunA == a = 0 /\ a' = 1 /\ UNCHANGED <<b, c>>

DoneA == a = 1 /\ a' = 2 /\ UNCHANGED <<b, c>>

RunB == b = 0 /\ a = 2 /\ b' = 1 /\ UNCHANGED <<a, c>>

DoneB == b = 1 /\ b' = 2 /\ UNCHANGED <<a, c>>

RunC == c = 0 /\ b = 2 /\ c' = 1 /\ UNCHANGED <<a, b>>

DoneC == c = 1 /\ c' = 2 /\ UNCHANGED <<a, b>>

Next == RunA \/ DoneA \/ RunB \/ DoneB \/ RunC \/ DoneC \/ UNCHANGED vars

Spec == Init /\ [][Next]_vars

TypeOK == a \in {0, 1, 2} /\ b \in {0, 1, 2} /\ c \in {0, 1, 2}

SafetyInvariant == (b > 0 => a = 2) /\ (c > 0 => b = 2)
====
"""))

# ---------------------------------------------------------------------------
# 6. Connection pool
# ---------------------------------------------------------------------------
SPECS.append(("manual4_connpool", "Model a connection pool with borrow and return operations. The pool has a fixed capacity and tracks available and in-use connections. Connections cannot exceed capacity.", r"""---- MODULE ConnPool ----
EXTENDS Integers

CONSTANT Max

VARIABLES avail, inuse

vars == <<avail, inuse>>

Init == avail = Max /\ inuse = 0

Borrow == avail > 0 /\ avail' = avail - 1 /\ inuse' = inuse + 1

Return == inuse > 0 /\ inuse' = inuse - 1 /\ avail' = avail + 1

Next == Borrow \/ Return \/ UNCHANGED vars

Spec == Init /\ [][Next]_vars

TypeOK == avail \in 0..Max /\ inuse \in 0..Max

SafetyConserved == avail + inuse = Max
====
"""))

# ---------------------------------------------------------------------------
# 7. Microservice health checker
# ---------------------------------------------------------------------------
SPECS.append(("manual4_healthcheck", "Model a microservice health checker that monitors service status. The service can be healthy, degraded, or down. A checker can probe, and the service can recover or degrade.", r"""---- MODULE HealthCheck ----
EXTENDS Integers

VARIABLES status, checks

vars == <<status, checks>>

Init == status = 2 /\ checks = 0

Degrade == status > 0 /\ status' = status - 1 /\ UNCHANGED checks

Recover == status < 2 /\ status' = status + 1 /\ UNCHANGED checks

Probe == checks < 3 /\ checks' = checks + 1 /\ UNCHANGED status

Next == Degrade \/ Recover \/ Probe \/ UNCHANGED vars

Spec == Init /\ [][Next]_vars

TypeOK == status \in {0, 1, 2} /\ checks \in 0..3

SafetyBounded == status >= 0 /\ status <= 2 /\ checks <= 3
====
"""))

# ---------------------------------------------------------------------------
# 8. Event sourcing (append events, bounded log)
# ---------------------------------------------------------------------------
SPECS.append(("manual4_eventsource", "Model an event sourcing system with a bounded event log. Events can be appended up to a maximum log size. A snapshot operation can compact the log.", r"""---- MODULE EventSource ----
EXTENDS Integers

CONSTANT Max

VARIABLES logLen, snapshots

vars == <<logLen, snapshots>>

Init == logLen = 0 /\ snapshots = 0

Append == logLen < Max /\ logLen' = logLen + 1 /\ UNCHANGED snapshots

Snapshot == logLen > 0 /\ snapshots < Max /\ snapshots' = snapshots + 1 /\ logLen' = 0

Next == Append \/ Snapshot \/ UNCHANGED vars

Spec == Init /\ [][Next]_vars

TypeOK == logLen \in 0..Max /\ snapshots \in 0..Max

SafetyBounded == logLen >= 0 /\ logLen <= Max
====
"""))

# ---------------------------------------------------------------------------
# 9. Feature flag toggle system
# ---------------------------------------------------------------------------
SPECS.append(("manual4_featureflag", "Model a feature flag system with multiple flags that can be toggled on or off. Track the number of enabled flags. The total number of flags is bounded.", r"""---- MODULE FeatureFlag ----
EXTENDS Integers

CONSTANT Max

VARIABLES enabled, total

vars == <<enabled, total>>

Init == enabled = 0 /\ total = Max

Toggle == enabled < total /\ enabled' = enabled + 1 /\ UNCHANGED total

Disable == enabled > 0 /\ enabled' = enabled - 1 /\ UNCHANGED total

Next == Toggle \/ Disable \/ UNCHANGED vars

Spec == Init /\ [][Next]_vars

TypeOK == enabled \in 0..Max /\ total \in 0..Max

SafetyBounded == enabled >= 0 /\ enabled <= total
====
"""))

# ---------------------------------------------------------------------------
# 10. A/B test assignment
# ---------------------------------------------------------------------------
SPECS.append(("manual4_abtest", "Model an A/B test assignment system that assigns users to group A or group B. The total number of assigned users is bounded. Track counts in each group.", r"""---- MODULE AbTest ----
EXTENDS Integers

CONSTANT Max

VARIABLES groupA, groupB

vars == <<groupA, groupB>>

Init == groupA = 0 /\ groupB = 0

AssignA == groupA + groupB < Max /\ groupA' = groupA + 1 /\ UNCHANGED groupB

AssignB == groupA + groupB < Max /\ groupB' = groupB + 1 /\ UNCHANGED groupA

Next == AssignA \/ AssignB \/ UNCHANGED vars

Spec == Init /\ [][Next]_vars

TypeOK == groupA \in 0..Max /\ groupB \in 0..Max

SafetyBounded == groupA + groupB <= Max
====
"""))

# ---------------------------------------------------------------------------
# 11. Batch processor
# ---------------------------------------------------------------------------
SPECS.append(("manual4_batchproc", "Model a batch processor that collects items into a buffer and processes them as a batch when full or triggered. The buffer size is bounded.", r"""---- MODULE BatchProc ----
EXTENDS Integers

CONSTANT Max

VARIABLES buf, processed

vars == <<buf, processed>>

Init == buf = 0 /\ processed = 0

Collect == buf < Max /\ buf' = buf + 1 /\ UNCHANGED processed

Flush == buf > 0 /\ processed < Max /\ processed' = processed + 1 /\ buf' = 0

Next == Collect \/ Flush \/ UNCHANGED vars

Spec == Init /\ [][Next]_vars

TypeOK == buf \in 0..Max /\ processed \in 0..Max

SafetyBounded == buf >= 0 /\ buf <= Max /\ processed >= 0 /\ processed <= Max
====
"""))

# ---------------------------------------------------------------------------
# 12. Service mesh circuit state
# ---------------------------------------------------------------------------
SPECS.append(("manual4_circuit", "Model a circuit breaker in a service mesh with three states: closed (normal), open (tripped), and half-open (probing). Track failure count that triggers state transitions.", r"""---- MODULE CircuitBreaker ----
EXTENDS Integers

VARIABLES state, failures

vars == <<state, failures>>

Init == state = 0 /\ failures = 0

Fail == state = 0 /\ failures < 3 /\ failures' = failures + 1 /\
        IF failures + 1 = 3 THEN state' = 1 ELSE state' = state

Succeed == state = 0 /\ failures > 0 /\ failures' = 0 /\ UNCHANGED state

Probe == state = 1 /\ state' = 2 /\ UNCHANGED failures

ProbeOk == state = 2 /\ state' = 0 /\ failures' = 0

ProbeFail == state = 2 /\ state' = 1 /\ UNCHANGED failures

Next == Fail \/ Succeed \/ Probe \/ ProbeOk \/ ProbeFail \/ UNCHANGED vars

Spec == Init /\ [][Next]_vars

TypeOK == state \in {0, 1, 2} /\ failures \in 0..3

SafetyInvariant == (state = 1 => failures = 3) /\ (state = 2 => failures \in {0, 3})
====
"""))

# ---------------------------------------------------------------------------
# 13. Quota management
# ---------------------------------------------------------------------------
SPECS.append(("manual4_quota", "Model a quota management system where quota units can be allocated and released. The total allocation cannot exceed the quota limit.", r"""---- MODULE QuotaMgmt ----
EXTENDS Integers

CONSTANT Max

VARIABLES allocated, remaining

vars == <<allocated, remaining>>

Init == allocated = 0 /\ remaining = Max

Allocate == remaining > 0 /\ allocated' = allocated + 1 /\ remaining' = remaining - 1

Release == allocated > 0 /\ allocated' = allocated - 1 /\ remaining' = remaining + 1

Next == Allocate \/ Release \/ UNCHANGED vars

Spec == Init /\ [][Next]_vars

TypeOK == allocated \in 0..Max /\ remaining \in 0..Max

SafetyConserved == allocated + remaining = Max
====
"""))

# ---------------------------------------------------------------------------
# 14. Canary deployment
# ---------------------------------------------------------------------------
SPECS.append(("manual4_canary", "Model a canary deployment system that gradually shifts traffic from a stable version to a canary version. Traffic percentage for the canary increases or rolls back. Total traffic is conserved.", r"""---- MODULE CanaryDeploy ----
EXTENDS Integers

CONSTANT Max

VARIABLES stable, canary

vars == <<stable, canary>>

Init == stable = Max /\ canary = 0

Shift == stable > 0 /\ stable' = stable - 1 /\ canary' = canary + 1

Rollback == canary > 0 /\ canary' = canary - 1 /\ stable' = stable + 1

Next == Shift \/ Rollback \/ UNCHANGED vars

Spec == Init /\ [][Next]_vars

TypeOK == stable \in 0..Max /\ canary \in 0..Max

SafetyConserved == stable + canary = Max
====
"""))

# ---------------------------------------------------------------------------
# 15. Garbage collector (mark/sweep simplified)
# ---------------------------------------------------------------------------
SPECS.append(("manual4_gc", "Model a simplified mark-and-sweep garbage collector. Objects can be allocated, marked as reachable, or swept (freed). Track allocated and marked counts with bounded heap.", r"""---- MODULE GarbageCollect ----
EXTENDS Integers

CONSTANT Max

VARIABLES alloc, marked, phase

vars == <<alloc, marked, phase>>

Init == alloc = 0 /\ marked = 0 /\ phase = 0

AllocObj == phase = 0 /\ alloc < Max /\ alloc' = alloc + 1 /\ UNCHANGED <<marked, phase>>

StartMark == phase = 0 /\ alloc > 0 /\ phase' = 1 /\ UNCHANGED <<alloc, marked>>

MarkObj == phase = 1 /\ marked < alloc /\ marked' = marked + 1 /\ UNCHANGED <<alloc, phase>>

StartSweep == phase = 1 /\ phase' = 2 /\ UNCHANGED <<alloc, marked>>

Sweep == phase = 2 /\ alloc' = marked /\ marked' = 0 /\ phase' = 0

Next == AllocObj \/ StartMark \/ MarkObj \/ StartSweep \/ Sweep \/ UNCHANGED vars

Spec == Init /\ [][Next]_vars

TypeOK == alloc \in 0..Max /\ marked \in 0..Max /\ phase \in {0, 1, 2}

SafetyBounded == marked <= alloc /\ alloc <= Max
====
"""))


def validate_and_save(do_validate=True, do_save=False):
    from scripts.diamond_sft_gen import validate_diamond, save_diamond_sft
    results = []
    for pid, prompt_text, spec in SPECS:
        if do_validate:
            r = validate_diamond(spec, prompt_id=pid, prompt_text=prompt_text, model="manual")
            results.append(r)
            status = "DIAMOND" if r.is_diamond else f"FAIL({r.error[:60]})"
            log.info(f"  {pid}: {status} states={r.distinct_states} inv={r.invariants_checked} mut={r.mutation_caught}")
        else:
            results.append(None)
    diamond_count = sum(1 for r in results if r and r.is_diamond)
    log.info(f"\nResults: {diamond_count}/{len(SPECS)} Diamond")
    if do_save and diamond_count > 0:
        diamond_results = [r for r in results if r and r.is_diamond]
        save_diamond_sft(diamond_results, append=True)
    return results


if __name__ == "__main__":
    do_save = "--save" in sys.argv
    validate_and_save(do_validate=True, do_save=do_save)
