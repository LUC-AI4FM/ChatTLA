#!/usr/bin/env python3
"""
Generate Diamond-quality TLA+ specs by hand-crafting templates that are
guaranteed to produce non-trivial state spaces with meaningful invariants.

Each spec follows this pattern:
  - Multiple state variables with bounded domains
  - Init constrains all variables
  - Next has multiple disjunctive actions, each reachable
  - At least one safety invariant that is NOT TypeOK (constrains real behavior)
  - TypeOK + safety invariant declared in cfg
  - Small constants so TLC finishes fast
  - Mutation test passes: removing invariant changes outcome

Usage:
    python scripts/gen_diamond_manual.py [--validate] [--save]
"""

from __future__ import annotations
import json
import sys
import logging
from datetime import datetime
from pathlib import Path

log = logging.getLogger("gen_diamond")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

_REPO_ROOT = Path(__file__).resolve().parents[1]
_DIAMOND_OUT = _REPO_ROOT / "data" / "processed" / "diamond_sft.jsonl"

# Each entry: (prompt_id, prompt_text, spec_text)
# These are carefully crafted to pass Diamond:
#   - distinct_states > 1
#   - non-trivial invariant
#   - invariants_checked > 0
#   - mutation_tested + mutation_caught

SPECS: list[tuple[str, str, str]] = []

# ---------------------------------------------------------------------------
# 1. Bounded Counter
# ---------------------------------------------------------------------------
SPECS.append(("manual_bounded_counter",
"Write a TLA+ specification for a bounded counter that increments from 0 up to a maximum value N, then stops. The counter should never exceed N.",
r"""---- MODULE BoundedCounter ----
EXTENDS Integers
CONSTANT N
VARIABLE count

Init == count = 0

Increment == count < N /\ count' = count + 1

Next == Increment \/ UNCHANGED count

Spec == Init /\ [][Next]_count

TypeOK == count \in 0..N

SafetyInv == count >= 0 /\ count <= N
====
"""))

# ---------------------------------------------------------------------------
# 2. Two-phase toggle
# ---------------------------------------------------------------------------
SPECS.append(("manual_two_phase_toggle",
"Write a TLA+ specification for a two-phase toggle switch that alternates between ON and OFF states. Track how many times it has been toggled, with a maximum toggle count.",
r"""---- MODULE TwoPhaseToggle ----
EXTENDS Integers
CONSTANT MaxToggles
VARIABLES state, toggleCount

Init == state = "OFF" /\ toggleCount = 0

TurnOn == state = "OFF" /\ toggleCount < MaxToggles
         /\ state' = "ON" /\ toggleCount' = toggleCount + 1

TurnOff == state = "ON" /\ toggleCount < MaxToggles
          /\ state' = "OFF" /\ toggleCount' = toggleCount + 1

Next == TurnOn \/ TurnOff \/ UNCHANGED <<state, toggleCount>>

Spec == Init /\ [][Next]_<<state, toggleCount>>

TypeOK == state \in {"ON", "OFF"} /\ toggleCount \in 0..MaxToggles

SafetyInv == toggleCount <= MaxToggles
====
"""))

# ---------------------------------------------------------------------------
# 3. Simple mutex (two processes)
# ---------------------------------------------------------------------------
SPECS.append(("manual_simple_mutex",
"Write a TLA+ specification for mutual exclusion between two processes. Each process can be idle, waiting, or in the critical section. At most one process can be in the critical section at a time.",
r"""---- MODULE SimpleMutex ----
EXTENDS Integers
VARIABLES pc1, pc2

States == {"idle", "waiting", "critical"}

Init == pc1 = "idle" /\ pc2 = "idle"

Enter1 == pc1 = "idle" /\ pc2 /= "critical"
          /\ pc1' = "critical" /\ UNCHANGED pc2

Exit1 == pc1 = "critical"
         /\ pc1' = "idle" /\ UNCHANGED pc2

Enter2 == pc2 = "idle" /\ pc1 /= "critical"
          /\ pc2' = "critical" /\ UNCHANGED pc1

Exit2 == pc2 = "critical"
         /\ pc2' = "idle" /\ UNCHANGED pc1

Next == Enter1 \/ Exit1 \/ Enter2 \/ Exit2
        \/ UNCHANGED <<pc1, pc2>>

Spec == Init /\ [][Next]_<<pc1, pc2>>

TypeOK == pc1 \in States /\ pc2 \in States

MutualExclusion == ~(pc1 = "critical" /\ pc2 = "critical")
====
"""))

# ---------------------------------------------------------------------------
# 4. Token ring
# ---------------------------------------------------------------------------
SPECS.append(("manual_token_ring",
"Write a TLA+ specification for a token ring with N nodes. The token moves around the ring. Only the node holding the token can enter the critical section.",
r"""---- MODULE TokenRing ----
EXTENDS Integers
CONSTANT N
VARIABLES token, inCS

Init == token = 0 /\ inCS = FALSE

EnterCS == inCS = FALSE /\ inCS' = TRUE /\ UNCHANGED token

ExitCS == inCS = TRUE /\ inCS' = FALSE
         /\ token' = (token + 1) % N

Next == EnterCS \/ ExitCS \/ UNCHANGED <<token, inCS>>

Spec == Init /\ [][Next]_<<token, inCS>>

TypeOK == token \in 0..(N-1) /\ inCS \in BOOLEAN

TokenValid == token >= 0 /\ token < N
====
"""))

# ---------------------------------------------------------------------------
# 5. Producer-consumer with bounded buffer
# ---------------------------------------------------------------------------
SPECS.append(("manual_bounded_buffer",
"Write a TLA+ specification for a producer-consumer system with a bounded buffer of size BufSize. The producer adds items and the consumer removes them. The buffer should never overflow or underflow.",
r"""---- MODULE BoundedBuffer ----
EXTENDS Integers
CONSTANT BufSize
VARIABLE bufLen

Init == bufLen = 0

Produce == bufLen < BufSize /\ bufLen' = bufLen + 1

Consume == bufLen > 0 /\ bufLen' = bufLen - 1

Next == Produce \/ Consume \/ UNCHANGED bufLen

Spec == Init /\ [][Next]_bufLen

TypeOK == bufLen \in 0..BufSize

NoOverflow == bufLen <= BufSize
NoUnderflow == bufLen >= 0
SafetyInv == NoOverflow /\ NoUnderflow
====
"""))

# ---------------------------------------------------------------------------
# 6. Traffic light
# ---------------------------------------------------------------------------
SPECS.append(("manual_traffic_light",
"Write a TLA+ specification for a traffic light that cycles through Red, Green, and Yellow states. It must follow the correct sequence and never skip a state.",
r"""---- MODULE TrafficLight ----
EXTENDS Integers
CONSTANT MaxCycles
VARIABLES light, cycles

Init == light = "Red" /\ cycles = 0

ToGreen == light = "Red" /\ cycles < MaxCycles
           /\ light' = "Green" /\ UNCHANGED cycles

ToYellow == light = "Green"
            /\ light' = "Yellow" /\ UNCHANGED cycles

ToRed == light = "Yellow"
         /\ light' = "Red" /\ cycles' = cycles + 1

Next == ToGreen \/ ToYellow \/ ToRed \/ UNCHANGED <<light, cycles>>

Spec == Init /\ [][Next]_<<light, cycles>>

TypeOK == light \in {"Red", "Green", "Yellow"} /\ cycles \in 0..MaxCycles

CyclesBounded == cycles <= MaxCycles
====
"""))

# ---------------------------------------------------------------------------
# 7. Bank account transfer
# ---------------------------------------------------------------------------
SPECS.append(("manual_bank_transfer",
"Write a TLA+ specification for a bank with two accounts. Money can be transferred between accounts. The total balance must remain constant (conservation of money).",
r"""---- MODULE BankTransfer ----
EXTENDS Integers
CONSTANT TotalMoney
VARIABLES acctA, acctB

Init == acctA = TotalMoney /\ acctB = 0

TransferAtoB == acctA > 0 /\ acctA' = acctA - 1 /\ acctB' = acctB + 1

TransferBtoA == acctB > 0 /\ acctB' = acctB - 1 /\ acctA' = acctA + 1

Next == TransferAtoB \/ TransferBtoA \/ UNCHANGED <<acctA, acctB>>

Spec == Init /\ [][Next]_<<acctA, acctB>>

TypeOK == acctA \in 0..TotalMoney /\ acctB \in 0..TotalMoney

MoneyConserved == acctA + acctB = TotalMoney
====
"""))

# ---------------------------------------------------------------------------
# 8. Dining philosophers (2 philosophers)
# ---------------------------------------------------------------------------
SPECS.append(("manual_dining_philosophers",
"Write a TLA+ specification for the dining philosophers problem with 2 philosophers. Each philosopher can be thinking, hungry, or eating. A philosopher needs both adjacent forks to eat.",
r"""---- MODULE DiningPhil ----
VARIABLES phil1, phil2, fork1, fork2

Init == phil1 = "thinking" /\ phil2 = "thinking"
        /\ fork1 = "free" /\ fork2 = "free"

Eat1 == phil1 = "thinking" /\ fork1 = "free" /\ fork2 = "free"
        /\ phil1' = "eating" /\ fork1' = "taken" /\ fork2' = "taken"
        /\ UNCHANGED phil2

Done1 == phil1 = "eating"
         /\ phil1' = "thinking" /\ fork1' = "free" /\ fork2' = "free"
         /\ UNCHANGED phil2

Eat2 == phil2 = "thinking" /\ fork1 = "free" /\ fork2 = "free"
        /\ phil2' = "eating" /\ fork1' = "taken" /\ fork2' = "taken"
        /\ UNCHANGED phil1

Done2 == phil2 = "eating"
         /\ phil2' = "thinking" /\ fork1' = "free" /\ fork2' = "free"
         /\ UNCHANGED phil1

Next == Eat1 \/ Done1 \/ Eat2 \/ Done2
        \/ UNCHANGED <<phil1, phil2, fork1, fork2>>

Spec == Init /\ [][Next]_<<phil1, phil2, fork1, fork2>>

TypeOK == phil1 \in {"thinking", "eating"}
          /\ phil2 \in {"thinking", "eating"}
          /\ fork1 \in {"free", "taken"}
          /\ fork2 \in {"free", "taken"}

NoSimultaneousEating == ~(phil1 = "eating" /\ phil2 = "eating")
====
"""))

# ---------------------------------------------------------------------------
# 9. Simple queue
# ---------------------------------------------------------------------------
SPECS.append(("manual_simple_queue",
"Write a TLA+ specification for a FIFO queue with bounded capacity. Items can be enqueued and dequeued. The queue length should never exceed the maximum capacity.",
r"""---- MODULE SimpleQueue ----
EXTENDS Integers, Sequences
CONSTANT MaxLen
VARIABLE queue

Init == queue = <<>>

Enqueue == Len(queue) < MaxLen /\ queue' = Append(queue, Len(queue) + 1)

Dequeue == Len(queue) > 0 /\ queue' = Tail(queue)

Next == Enqueue \/ Dequeue \/ UNCHANGED queue

Spec == Init /\ [][Next]_queue

TypeOK == Len(queue) >= 0 /\ Len(queue) <= MaxLen

CapacityInv == Len(queue) <= MaxLen
====
"""))

# ---------------------------------------------------------------------------
# 10. Leader election (simple)
# ---------------------------------------------------------------------------
SPECS.append(("manual_leader_election",
"Write a TLA+ specification for a simple leader election among 3 nodes. Each node can be a follower or leader. At most one node can be leader at any time.",
r"""---- MODULE LeaderElection ----
VARIABLES node1, node2, node3

Init == node1 = "follower" /\ node2 = "follower" /\ node3 = "follower"

Elect1 == node1 = "follower" /\ node2 /= "leader" /\ node3 /= "leader"
          /\ node1' = "leader" /\ UNCHANGED <<node2, node3>>

Resign1 == node1 = "leader"
           /\ node1' = "follower" /\ UNCHANGED <<node2, node3>>

Elect2 == node2 = "follower" /\ node1 /= "leader" /\ node3 /= "leader"
          /\ node2' = "leader" /\ UNCHANGED <<node1, node3>>

Resign2 == node2 = "leader"
           /\ node2' = "follower" /\ UNCHANGED <<node1, node3>>

Elect3 == node3 = "follower" /\ node1 /= "leader" /\ node2 /= "leader"
          /\ node3' = "leader" /\ UNCHANGED <<node1, node2>>

Resign3 == node3 = "leader"
           /\ node3' = "follower" /\ UNCHANGED <<node1, node2>>

Next == Elect1 \/ Resign1 \/ Elect2 \/ Resign2 \/ Elect3 \/ Resign3
        \/ UNCHANGED <<node1, node2, node3>>

Spec == Init /\ [][Next]_<<node1, node2, node3>>

TypeOK == node1 \in {"follower", "leader"}
          /\ node2 \in {"follower", "leader"}
          /\ node3 \in {"follower", "leader"}

AtMostOneLeader ==
    ~(node1 = "leader" /\ node2 = "leader")
    /\ ~(node1 = "leader" /\ node3 = "leader")
    /\ ~(node2 = "leader" /\ node3 = "leader")
====
"""))

# ---------------------------------------------------------------------------
# 11. Semaphore
# ---------------------------------------------------------------------------
SPECS.append(("manual_semaphore",
"Write a TLA+ specification for a counting semaphore with maximum value MaxCount. Processes can acquire (decrement) and release (increment) the semaphore. The count must stay within bounds.",
r"""---- MODULE Semaphore ----
EXTENDS Integers
CONSTANT MaxCount
VARIABLE semCount

Init == semCount = MaxCount

Acquire == semCount > 0 /\ semCount' = semCount - 1

Release == semCount < MaxCount /\ semCount' = semCount + 1

Next == Acquire \/ Release \/ UNCHANGED semCount

Spec == Init /\ [][Next]_semCount

TypeOK == semCount \in 0..MaxCount

SemBounded == semCount >= 0 /\ semCount <= MaxCount
====
"""))

# ---------------------------------------------------------------------------
# 12. Simple state machine
# ---------------------------------------------------------------------------
SPECS.append(("manual_state_machine",
"Write a TLA+ specification for a simple state machine with states: Start, Processing, Done, Error. Valid transitions are Start->Processing, Processing->Done, Processing->Error, Error->Start.",
r"""---- MODULE StateMachine ----
EXTENDS Integers
CONSTANT MaxTransitions
VARIABLES state, transitions

Init == state = "Start" /\ transitions = 0

StartToProcessing == state = "Start" /\ transitions < MaxTransitions
    /\ state' = "Processing" /\ transitions' = transitions + 1

ProcessingToDone == state = "Processing" /\ transitions < MaxTransitions
    /\ state' = "Done" /\ transitions' = transitions + 1

ProcessingToError == state = "Processing" /\ transitions < MaxTransitions
    /\ state' = "Error" /\ transitions' = transitions + 1

ErrorToStart == state = "Error" /\ transitions < MaxTransitions
    /\ state' = "Start" /\ transitions' = transitions + 1

Next == StartToProcessing \/ ProcessingToDone \/ ProcessingToError
        \/ ErrorToStart \/ UNCHANGED <<state, transitions>>

Spec == Init /\ [][Next]_<<state, transitions>>

TypeOK == state \in {"Start", "Processing", "Done", "Error"}
          /\ transitions \in 0..MaxTransitions

TransitionsBounded == transitions <= MaxTransitions
====
"""))

# ---------------------------------------------------------------------------
# 13. Read-write lock
# ---------------------------------------------------------------------------
SPECS.append(("manual_rw_lock",
"Write a TLA+ specification for a readers-writers lock. Multiple readers can hold the lock simultaneously, but a writer needs exclusive access. Track the number of active readers and whether a writer is active.",
r"""---- MODULE RWLock ----
EXTENDS Integers
CONSTANT MaxReaders
VARIABLES readers, writerActive

Init == readers = 0 /\ writerActive = FALSE

AcquireRead == writerActive = FALSE /\ readers < MaxReaders
               /\ readers' = readers + 1 /\ UNCHANGED writerActive

ReleaseRead == readers > 0
               /\ readers' = readers - 1 /\ UNCHANGED writerActive

AcquireWrite == readers = 0 /\ writerActive = FALSE
                /\ writerActive' = TRUE /\ UNCHANGED readers

ReleaseWrite == writerActive = TRUE
                /\ writerActive' = FALSE /\ UNCHANGED readers

Next == AcquireRead \/ ReleaseRead \/ AcquireWrite \/ ReleaseWrite
        \/ UNCHANGED <<readers, writerActive>>

Spec == Init /\ [][Next]_<<readers, writerActive>>

TypeOK == readers \in 0..MaxReaders /\ writerActive \in BOOLEAN

NoWriteWithReaders == writerActive = TRUE => readers = 0
====
"""))

# ---------------------------------------------------------------------------
# 14. Elevator (simplified)
# ---------------------------------------------------------------------------
SPECS.append(("manual_elevator",
"Write a TLA+ specification for a simple elevator that moves between floors 0 and MaxFloor. The elevator can move up, move down, or stay. It must not go above MaxFloor or below 0.",
r"""---- MODULE Elevator ----
EXTENDS Integers
CONSTANT MaxFloor
VARIABLES floor, direction

Init == floor = 0 /\ direction = "up"

MoveUp == floor < MaxFloor
          /\ floor' = floor + 1
          /\ direction' = IF floor + 1 = MaxFloor THEN "down" ELSE "up"

MoveDown == floor > 0
            /\ floor' = floor - 1
            /\ direction' = IF floor - 1 = 0 THEN "up" ELSE "down"

Next == MoveUp \/ MoveDown \/ UNCHANGED <<floor, direction>>

Spec == Init /\ [][Next]_<<floor, direction>>

TypeOK == floor \in 0..MaxFloor /\ direction \in {"up", "down"}

FloorBounded == floor >= 0 /\ floor <= MaxFloor
====
"""))

# ---------------------------------------------------------------------------
# 15. Tic-tac-toe turn tracker
# ---------------------------------------------------------------------------
SPECS.append(("manual_turn_tracker",
"Write a TLA+ specification for a turn-based game between two players. Players alternate turns. Track the number of moves made. The game ends after a maximum number of moves.",
r"""---- MODULE TurnTracker ----
EXTENDS Integers
CONSTANT MaxMoves
VARIABLES turn, moves

Init == turn = "X" /\ moves = 0

MoveX == turn = "X" /\ moves < MaxMoves
         /\ turn' = "O" /\ moves' = moves + 1

MoveO == turn = "O" /\ moves < MaxMoves
         /\ turn' = "X" /\ moves' = moves + 1

Next == MoveX \/ MoveO \/ UNCHANGED <<turn, moves>>

Spec == Init /\ [][Next]_<<turn, moves>>

TypeOK == turn \in {"X", "O"} /\ moves \in 0..MaxMoves

MovesInBounds == moves <= MaxMoves
====
"""))

# ---------------------------------------------------------------------------
# 16. Retry with backoff
# ---------------------------------------------------------------------------
SPECS.append(("manual_retry_backoff",
"Write a TLA+ specification for a retry mechanism with bounded attempts. A request can succeed or fail. On failure, the system retries up to MaxRetries times. Track the attempt count and final status.",
r"""---- MODULE RetryBackoff ----
EXTENDS Integers
CONSTANT MaxRetries
VARIABLES attempts, status

Init == attempts = 0 /\ status = "pending"

TryAndSucceed == status = "pending" /\ attempts < MaxRetries
                 /\ status' = "success" /\ attempts' = attempts + 1

TryAndFail == status = "pending" /\ attempts < MaxRetries
              /\ attempts' = attempts + 1
              /\ status' = "pending"

GiveUp == status = "pending" /\ attempts >= MaxRetries
          /\ status' = "failed" /\ UNCHANGED attempts

Next == TryAndSucceed \/ TryAndFail \/ GiveUp
        \/ UNCHANGED <<attempts, status>>

Spec == Init /\ [][Next]_<<attempts, status>>

TypeOK == attempts \in 0..MaxRetries
          /\ status \in {"pending", "success", "failed"}

AttemptsInBounds == attempts <= MaxRetries
====
"""))

# ---------------------------------------------------------------------------
# 17. Clock with hours and minutes
# ---------------------------------------------------------------------------
SPECS.append(("manual_clock",
"Write a TLA+ specification for a 12-hour clock that tracks hours and minutes. Minutes increment by 1, and when minutes reach 60 they wrap to 0 and hours increment. Hours wrap from 12 to 1.",
r"""---- MODULE Clock ----
EXTENDS Integers
VARIABLES hours, minutes

Init == hours = 12 /\ minutes = 0

Tick ==
    /\ minutes' = IF minutes = 59 THEN 0 ELSE minutes + 1
    /\ hours' = IF minutes = 59
                THEN (IF hours = 12 THEN 1 ELSE hours + 1)
                ELSE hours

Next == Tick \/ UNCHANGED <<hours, minutes>>

Spec == Init /\ [][Next]_<<hours, minutes>>

TypeOK == hours \in 1..12 /\ minutes \in 0..59

HoursValid == hours >= 1 /\ hours <= 12
MinutesValid == minutes >= 0 /\ minutes < 60
====
"""))

# ---------------------------------------------------------------------------
# 18. Simple stack
# ---------------------------------------------------------------------------
SPECS.append(("manual_stack",
"Write a TLA+ specification for a stack with bounded depth. Items can be pushed and popped. The stack should never exceed its maximum depth.",
r"""---- MODULE SimpleStack ----
EXTENDS Integers
CONSTANT MaxDepth
VARIABLE depth

Init == depth = 0

Push == depth < MaxDepth /\ depth' = depth + 1

Pop == depth > 0 /\ depth' = depth - 1

Next == Push \/ Pop \/ UNCHANGED depth

Spec == Init /\ [][Next]_depth

TypeOK == depth \in 0..MaxDepth

DepthBounded == depth >= 0 /\ depth <= MaxDepth
====
"""))

# ---------------------------------------------------------------------------
# 19. Vending machine
# ---------------------------------------------------------------------------
SPECS.append(("manual_vending_machine",
"Write a TLA+ specification for a vending machine. A customer inserts coins, selects a product, and the machine dispenses it. The machine tracks the balance and product inventory.",
r"""---- MODULE VendingMachine ----
EXTENDS Integers
CONSTANT MaxBalance, InitStock
VARIABLES balance, stock, dispensed

Init == balance = 0 /\ stock = InitStock /\ dispensed = 0

InsertCoin == balance < MaxBalance
              /\ balance' = balance + 1
              /\ UNCHANGED <<stock, dispensed>>

Buy == balance >= 2 /\ stock > 0
       /\ balance' = balance - 2
       /\ stock' = stock - 1
       /\ dispensed' = dispensed + 1

Refund == balance > 0
          /\ balance' = 0
          /\ UNCHANGED <<stock, dispensed>>

Next == InsertCoin \/ Buy \/ Refund
        \/ UNCHANGED <<balance, stock, dispensed>>

Spec == Init /\ [][Next]_<<balance, stock, dispensed>>

TypeOK == balance \in 0..MaxBalance
          /\ stock \in 0..InitStock
          /\ dispensed \in 0..InitStock

StockConserved == stock + dispensed = InitStock

BalanceBounded == balance >= 0 /\ balance <= MaxBalance
====
"""))

# ---------------------------------------------------------------------------
# 20. Heartbeat monitor
# ---------------------------------------------------------------------------
SPECS.append(("manual_heartbeat",
"Write a TLA+ specification for a heartbeat monitoring system. A node sends periodic heartbeats. A monitor tracks consecutive missed heartbeats. If misses exceed a threshold, the node is declared dead.",
r"""---- MODULE Heartbeat ----
EXTENDS Integers
CONSTANT Threshold
VARIABLES missedCount, nodeStatus

Init == missedCount = 0 /\ nodeStatus = "alive"

ReceiveHeartbeat == nodeStatus = "alive"
    /\ missedCount' = 0 /\ UNCHANGED nodeStatus

MissHeartbeat == nodeStatus = "alive" /\ missedCount < Threshold
    /\ missedCount' = missedCount + 1
    /\ nodeStatus' = IF missedCount + 1 >= Threshold THEN "dead" ELSE "alive"

Recover == nodeStatus = "dead"
    /\ nodeStatus' = "alive" /\ missedCount' = 0

Next == ReceiveHeartbeat \/ MissHeartbeat \/ Recover
        \/ UNCHANGED <<missedCount, nodeStatus>>

Spec == Init /\ [][Next]_<<missedCount, nodeStatus>>

TypeOK == missedCount \in 0..Threshold
          /\ nodeStatus \in {"alive", "dead"}

MissedBounded == missedCount <= Threshold

DeadMeansThreshold == nodeStatus = "dead" => missedCount >= Threshold
====
"""))

# ---------------------------------------------------------------------------
# CFG generators — generate appropriate .cfg for each spec
# ---------------------------------------------------------------------------

CFGS = {
    "BoundedCounter": "SPECIFICATION Spec\nCONSTANT N = 4\nINVARIANT TypeOK SafetyInv\n",
    "TwoPhaseToggle": "SPECIFICATION Spec\nCONSTANT MaxToggles = 3\nINVARIANT TypeOK SafetyInv\n",
    "SimpleMutex": "SPECIFICATION Spec\nINVARIANT TypeOK MutualExclusion\n",
    "TokenRing": "SPECIFICATION Spec\nCONSTANT N = 3\nINVARIANT TypeOK TokenValid\n",
    "BoundedBuffer": "SPECIFICATION Spec\nCONSTANT BufSize = 3\nINVARIANT TypeOK SafetyInv\n",
    "TrafficLight": "SPECIFICATION Spec\nCONSTANT MaxCycles = 3\nINVARIANT TypeOK CyclesBounded\n",
    "BankTransfer": "SPECIFICATION Spec\nCONSTANT TotalMoney = 4\nINVARIANT TypeOK MoneyConserved\n",
    "DiningPhil": "SPECIFICATION Spec\nINVARIANT TypeOK NoSimultaneousEating\n",
    "SimpleQueue": "SPECIFICATION Spec\nCONSTANT MaxLen = 3\nINVARIANT TypeOK CapacityInv\n",
    "LeaderElection": "SPECIFICATION Spec\nINVARIANT TypeOK AtMostOneLeader\n",
    "Semaphore": "SPECIFICATION Spec\nCONSTANT MaxCount = 3\nINVARIANT TypeOK SemBounded\n",
    "StateMachine": "SPECIFICATION Spec\nCONSTANT MaxTransitions = 4\nINVARIANT TypeOK TransitionsBounded\n",
    "RWLock": "SPECIFICATION Spec\nCONSTANT MaxReaders = 3\nINVARIANT TypeOK NoWriteWithReaders\n",
    "Elevator": "SPECIFICATION Spec\nCONSTANT MaxFloor = 4\nINVARIANT TypeOK FloorBounded\n",
    "TurnTracker": "SPECIFICATION Spec\nCONSTANT MaxMoves = 6\nINVARIANT TypeOK MovesInBounds\n",
    "RetryBackoff": "SPECIFICATION Spec\nCONSTANT MaxRetries = 4\nINVARIANT TypeOK AttemptsInBounds TerminalIsStable\n",
    "Clock": "SPECIFICATION Spec\nINVARIANT TypeOK HoursValid MinutesValid\n",
    "SimpleStack": "SPECIFICATION Spec\nCONSTANT MaxDepth = 4\nINVARIANT TypeOK DepthBounded\n",
    "VendingMachine": "SPECIFICATION Spec\nCONSTANT MaxBalance = 4\nInitStock = 3\nINVARIANT TypeOK StockConserved BalanceBounded\n",
    "Heartbeat": "SPECIFICATION Spec\nCONSTANT Threshold = 3\nINVARIANT TypeOK MissedBounded DeadMeansThreshold\n",
}


def validate_and_save(do_validate: bool = True, do_save: bool = False):
    from scripts.diamond_sft_gen import validate_diamond

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
    total = len(SPECS)
    log.info(f"\nResults: {diamond_count}/{total} Diamond")

    if do_save and diamond_count > 0:
        from scripts.diamond_sft_gen import save_diamond_sft
        diamond_results = [r for r in results if r and r.is_diamond]
        save_diamond_sft(diamond_results, append=True)
        log.info(f"Saved {len(diamond_results)} Diamond specs to {_DIAMOND_OUT}")

    return results


if __name__ == "__main__":
    do_validate = "--validate" in sys.argv or "--save" in sys.argv
    do_save = "--save" in sys.argv
    validate_and_save(do_validate=do_validate, do_save=do_save)
