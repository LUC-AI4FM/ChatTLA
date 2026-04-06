#!/usr/bin/env python3
"""Batch 3 of hand-crafted Diamond TLA+ specs."""
from __future__ import annotations
import json, logging, sys
from datetime import datetime
from pathlib import Path

log = logging.getLogger("gen_diamond_b3")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

_REPO_ROOT = Path(__file__).resolve().parents[1]
_DIAMOND_OUT = _REPO_ROOT / "data" / "processed" / "diamond_sft.jsonl"

SPECS: list[tuple[str, str, str]] = []

# Each entry: (prompt_id, prompt_text, spec_text)

# ---------------------------------------------------------------------------
# 1. Network packet router with bounded queue
# ---------------------------------------------------------------------------
SPECS.append(("manual3_packet_router",
"Model a network packet router with a bounded queue. Packets arrive and are forwarded. The queue has a maximum capacity and must never overflow.",
r"""---- MODULE PacketRouter ----
EXTENDS Integers
CONSTANT Max
VARIABLE queue

vars == <<queue>>

Init == queue = 0

Enqueue == queue < Max /\ queue' = queue + 1
Dequeue == queue > 0 /\ queue' = queue - 1

Next == Enqueue \/ Dequeue \/ UNCHANGED vars

Spec == Init /\ [][Next]_vars

TypeOK == queue \in 0..Max
SafetyBounded == queue >= 0 /\ queue <= Max
====
"""))

# ---------------------------------------------------------------------------
# 2. Memory allocator (alloc/free blocks)
# ---------------------------------------------------------------------------
SPECS.append(("manual3_mem_alloc",
"Model a simple memory allocator that can allocate and free blocks from a fixed pool. Track allocated count and ensure it never exceeds capacity.",
r"""---- MODULE MemAlloc ----
EXTENDS Integers
CONSTANT Max
VARIABLE allocated

vars == <<allocated>>

Init == allocated = 0

Alloc == allocated < Max /\ allocated' = allocated + 1
Free  == allocated > 0 /\ allocated' = allocated - 1

Next == Alloc \/ Free \/ UNCHANGED vars

Spec == Init /\ [][Next]_vars

TypeOK == allocated \in 0..Max
SafetyBounded == allocated >= 0 /\ allocated <= Max
NoOverflow == allocated <= Max
====
"""))

# ---------------------------------------------------------------------------
# 3. Load balancer (round-robin with N servers)
# ---------------------------------------------------------------------------
SPECS.append(("manual3_load_balancer",
"Model a round-robin load balancer distributing requests across N servers. Track current server index and total requests dispatched.",
r"""---- MODULE LoadBalancer ----
EXTENDS Integers
CONSTANT N
VARIABLES current, dispatched

vars == <<current, dispatched>>

Init == current = 0 /\ dispatched = 0

Dispatch == /\ dispatched < N * N
            /\ current' = (current + 1) % N
            /\ dispatched' = dispatched + 1

Next == Dispatch \/ UNCHANGED vars

Spec == Init /\ [][Next]_vars

TypeOK == /\ current \in 0..(N-1)
          /\ dispatched \in 0..(N*N)
SafetyBounded == dispatched <= N * N
SafetyValid == current >= 0 /\ current < N
====
"""))

# ---------------------------------------------------------------------------
# 4. Publish-subscribe message broker
# ---------------------------------------------------------------------------
SPECS.append(("manual3_pubsub",
"Model a publish-subscribe message broker with a bounded message buffer. Publishers add messages, subscribers consume them. The buffer must not overflow.",
r"""---- MODULE PubSub ----
EXTENDS Integers
CONSTANT Max
VARIABLES pending, consumed

vars == <<pending, consumed>>

Init == pending = 0 /\ consumed = 0

Publish   == /\ pending < Max
             /\ pending' = pending + 1
             /\ UNCHANGED consumed

Subscribe == /\ pending > 0
             /\ consumed < Max
             /\ pending' = pending - 1
             /\ consumed' = consumed + 1

Reset == /\ consumed > 0
         /\ consumed' = consumed - 1
         /\ UNCHANGED pending

Next == Publish \/ Subscribe \/ Reset \/ UNCHANGED vars

Spec == Init /\ [][Next]_vars

TypeOK == /\ pending \in 0..Max
          /\ consumed \in 0..Max
SafetyBounded == pending >= 0 /\ pending <= Max
SafetyValid == consumed >= 0 /\ consumed <= Max
====
"""))

# ---------------------------------------------------------------------------
# 5. Simple blockchain (append-only chain with validation)
# ---------------------------------------------------------------------------
SPECS.append(("manual3_blockchain",
"Model a simplified blockchain where blocks are appended to a chain. The chain length is bounded. Track confirmed vs pending blocks.",
r"""---- MODULE SimpleChain ----
EXTENDS Integers
CONSTANT Max
VARIABLES pending, confirmed

vars == <<pending, confirmed>>

Init == pending = 0 /\ confirmed = 0

AddBlock == /\ pending + confirmed < Max
            /\ pending' = pending + 1
            /\ UNCHANGED confirmed

ConfirmBlock == /\ pending > 0
                /\ confirmed' = confirmed + 1
                /\ pending' = pending - 1

Next == AddBlock \/ ConfirmBlock \/ UNCHANGED vars

Spec == Init /\ [][Next]_vars

TypeOK == /\ pending \in 0..Max
          /\ confirmed \in 0..Max
SafetyBounded == pending + confirmed <= Max
SafetyValid == pending >= 0 /\ confirmed >= 0
====
"""))

# ---------------------------------------------------------------------------
# 6. Thread pool (fixed size, task submission)
# ---------------------------------------------------------------------------
SPECS.append(("manual3_thread_pool",
"Model a fixed-size thread pool that accepts task submissions. Track active threads and queued tasks. The pool must never exceed its capacity.",
r"""---- MODULE ThreadPool ----
EXTENDS Integers
CONSTANT Max
VARIABLES active, queued

vars == <<active, queued>>

Init == active = 0 /\ queued = 0

Submit   == queued < Max /\ queued' = queued + 1 /\ UNCHANGED active
StartTask == /\ queued > 0 /\ active < Max
             /\ active' = active + 1
             /\ queued' = queued - 1
FinishTask == active > 0 /\ active' = active - 1 /\ UNCHANGED queued

Next == Submit \/ StartTask \/ FinishTask \/ UNCHANGED vars

Spec == Init /\ [][Next]_vars

TypeOK == /\ active \in 0..Max
          /\ queued \in 0..Max
SafetyBounded == active <= Max /\ queued <= Max
SafetyValid == active >= 0 /\ queued >= 0
====
"""))

# ---------------------------------------------------------------------------
# 7. DNS resolver cache with TTL
# ---------------------------------------------------------------------------
SPECS.append(("manual3_dns_cache",
"Model a DNS resolver cache with TTL. Entries are cached with a time-to-live counter. They expire when TTL reaches zero and can be refreshed.",
r"""---- MODULE DnsCache ----
EXTENDS Integers
CONSTANT Max
VARIABLES entries, ttl

vars == <<entries, ttl>>

Init == entries = 0 /\ ttl = 0

CacheEntry == /\ entries < Max
              /\ entries' = entries + 1
              /\ ttl' = Max

TickTTL == /\ ttl > 0
           /\ ttl' = ttl - 1
           /\ UNCHANGED entries

Expire == /\ entries > 0 /\ ttl = 0
          /\ entries' = entries - 1
          /\ ttl' = 0

Refresh == /\ entries > 0
           /\ ttl' = Max
           /\ UNCHANGED entries

Next == CacheEntry \/ TickTTL \/ Expire \/ Refresh \/ UNCHANGED vars

Spec == Init /\ [][Next]_vars

TypeOK == /\ entries \in 0..Max
          /\ ttl \in 0..Max
SafetyBounded == entries <= Max /\ ttl <= Max
SafetyValid == entries >= 0 /\ ttl >= 0
====
"""))

# ---------------------------------------------------------------------------
# 8. Shopping cart (add/remove items, bounded quantity)
# ---------------------------------------------------------------------------
SPECS.append(("manual3_shopping_cart",
"Model a shopping cart where items can be added and removed. The cart has a maximum capacity. Track items in cart and items checked out.",
r"""---- MODULE ShoppingCart ----
EXTENDS Integers
CONSTANT Max
VARIABLES inCart, checkedOut

vars == <<inCart, checkedOut>>

Init == inCart = 0 /\ checkedOut = 0

AddItem    == /\ inCart < Max
              /\ inCart' = inCart + 1
              /\ UNCHANGED checkedOut

RemoveItem == /\ inCart > 0
              /\ inCart' = inCart - 1
              /\ UNCHANGED checkedOut

Checkout   == /\ inCart > 0 /\ checkedOut < Max
              /\ checkedOut' = checkedOut + 1
              /\ inCart' = inCart - 1

Next == AddItem \/ RemoveItem \/ Checkout \/ UNCHANGED vars

Spec == Init /\ [][Next]_vars

TypeOK == /\ inCart \in 0..Max
          /\ checkedOut \in 0..Max
SafetyBounded == inCart <= Max /\ checkedOut <= Max
SafetyValid == inCart >= 0 /\ checkedOut >= 0
====
"""))

# ---------------------------------------------------------------------------
# 9. Printer queue with priority levels
# ---------------------------------------------------------------------------
SPECS.append(("manual3_printer_queue",
"Model a printer queue with two priority levels (high and low). High-priority jobs are served first. Total jobs are bounded.",
r"""---- MODULE PrinterQueue ----
EXTENDS Integers
CONSTANT Max
VARIABLES high, low

vars == <<high, low>>

Init == high = 0 /\ low = 0

AddHigh == high + low < Max /\ high' = high + 1 /\ UNCHANGED low
AddLow  == high + low < Max /\ low' = low + 1 /\ UNCHANGED high
PrintHigh == high > 0 /\ high' = high - 1 /\ UNCHANGED low
PrintLow  == high = 0 /\ low > 0 /\ low' = low - 1 /\ UNCHANGED high

Next == AddHigh \/ AddLow \/ PrintHigh \/ PrintLow \/ UNCHANGED vars

Spec == Init /\ [][Next]_vars

TypeOK == /\ high \in 0..Max
          /\ low \in 0..Max
SafetyBounded == high + low <= Max
SafetyValid == high >= 0 /\ low >= 0
====
"""))

# ---------------------------------------------------------------------------
# 10. Power state machine (off/sleep/active/charging)
# ---------------------------------------------------------------------------
SPECS.append(("manual3_power_state",
"Model a device power state machine with states: off, sleep, active, and charging. Define valid transitions and ensure the device is always in a known state.",
r"""---- MODULE PowerState ----
EXTENDS Integers
VARIABLES state, battery

vars == <<state, battery>>

Init == state = 0 /\ battery = 2

\* States: 0=off, 1=sleep, 2=active, 3=charging

TurnOn    == state = 0 /\ battery > 0 /\ state' = 2 /\ UNCHANGED battery
Sleep     == state = 2 /\ state' = 1 /\ UNCHANGED battery
Wake      == state = 1 /\ battery > 0 /\ state' = 2 /\ UNCHANGED battery
TurnOff   == state \in {1, 2} /\ state' = 0 /\ UNCHANGED battery
StartCharge == state = 0 /\ battery < 3 /\ state' = 3 /\ UNCHANGED battery
Charge    == state = 3 /\ battery < 3 /\ battery' = battery + 1 /\ UNCHANGED state
StopCharge == state = 3 /\ state' = 0 /\ UNCHANGED battery
Drain     == state = 2 /\ battery > 0 /\ battery' = battery - 1 /\ UNCHANGED state

Next == TurnOn \/ Sleep \/ Wake \/ TurnOff \/ StartCharge \/ Charge \/ StopCharge \/ Drain \/ UNCHANGED vars

Spec == Init /\ [][Next]_vars

TypeOK == /\ state \in 0..3
          /\ battery \in 0..3
SafetyValid == state \in {0, 1, 2, 3}
SafetyBounded == battery >= 0 /\ battery <= 3
====
"""))

# ---------------------------------------------------------------------------
# 11. Raft-simplified log replication
# ---------------------------------------------------------------------------
SPECS.append(("manual3_raft_log",
"Model a simplified Raft-like log replication protocol. A leader appends entries to its log and followers replicate. Track commit index.",
r"""---- MODULE RaftLog ----
EXTENDS Integers
CONSTANT Max
VARIABLES leaderLog, followerLog, commitIdx

vars == <<leaderLog, followerLog, commitIdx>>

Init == leaderLog = 0 /\ followerLog = 0 /\ commitIdx = 0

AppendEntry == /\ leaderLog < Max
               /\ leaderLog' = leaderLog + 1
               /\ UNCHANGED <<followerLog, commitIdx>>

Replicate == /\ followerLog < leaderLog
             /\ followerLog' = followerLog + 1
             /\ UNCHANGED <<leaderLog, commitIdx>>

Commit == /\ commitIdx < followerLog
          /\ commitIdx' = commitIdx + 1
          /\ UNCHANGED <<leaderLog, followerLog>>

Next == AppendEntry \/ Replicate \/ Commit \/ UNCHANGED vars

Spec == Init /\ [][Next]_vars

TypeOK == /\ leaderLog \in 0..Max
          /\ followerLog \in 0..Max
          /\ commitIdx \in 0..Max
SafetyValid == /\ followerLog <= leaderLog
               /\ commitIdx <= followerLog
SafetyBounded == leaderLog <= Max
====
"""))

# ---------------------------------------------------------------------------
# 12. Water tank level controller
# ---------------------------------------------------------------------------
SPECS.append(("manual3_water_tank",
"Model a water tank level controller. Water can be added (fill) or drained. The level must stay within safe bounds (not overflow or underflow).",
r"""---- MODULE WaterTank ----
EXTENDS Integers
CONSTANT Max
VARIABLE level

vars == <<level>>

Init == level = 0

Fill  == level < Max /\ level' = level + 1
Drain == level > 0 /\ level' = level - 1

Next == Fill \/ Drain \/ UNCHANGED vars

Spec == Init /\ [][Next]_vars

TypeOK == level \in 0..Max
SafetyBounded == level >= 0 /\ level <= Max
NoOverflow == level <= Max
NoUnderflow == level >= 0
====
"""))

# ---------------------------------------------------------------------------
# 13. Parking lot management (enter/exit, bounded spaces)
# ---------------------------------------------------------------------------
SPECS.append(("manual3_parking_lot",
"Model a parking lot with bounded spaces and a waiting queue. Vehicles can enter when spaces are available, queue when full, and exit.",
r"""---- MODULE ParkingLot ----
EXTENDS Integers
CONSTANT Max
VARIABLES occupied, waiting

vars == <<occupied, waiting>>

Init == occupied = 0 /\ waiting = 0

Enter == /\ occupied < Max /\ waiting = 0
         /\ occupied' = occupied + 1
         /\ UNCHANGED waiting

QueueUp == /\ occupied = Max /\ waiting < Max
           /\ waiting' = waiting + 1
           /\ UNCHANGED occupied

Admit == /\ occupied < Max /\ waiting > 0
         /\ occupied' = occupied + 1
         /\ waiting' = waiting - 1

Exit == /\ occupied > 0
        /\ occupied' = occupied - 1
        /\ UNCHANGED waiting

Next == Enter \/ QueueUp \/ Admit \/ Exit \/ UNCHANGED vars

Spec == Init /\ [][Next]_vars

TypeOK == /\ occupied \in 0..Max
          /\ waiting \in 0..Max
SafetyBounded == occupied <= Max /\ waiting <= Max
SafetyValid == occupied >= 0 /\ waiting >= 0
====
"""))

# ---------------------------------------------------------------------------
# 14. Cruise control (accelerate/brake/maintain speed bounds)
# ---------------------------------------------------------------------------
SPECS.append(("manual3_cruise_ctrl",
"Model a cruise control system that can accelerate, brake, and maintain speed. Speed is bounded between 0 and a maximum value. Braking while engaged auto-disengages at speed 0.",
r"""---- MODULE CruiseControl ----
EXTENDS Integers
CONSTANT Max
VARIABLES speed, engaged

vars == <<speed, engaged>>

Init == speed = 0 /\ engaged = 0

Engage    == engaged = 0 /\ speed > 0 /\ engaged' = 1 /\ UNCHANGED speed
Disengage == engaged = 1 /\ engaged' = 0 /\ UNCHANGED speed
Accelerate == speed < Max /\ speed' = speed + 1 /\ UNCHANGED engaged
Brake      == /\ speed > 1
              /\ speed' = speed - 1
              /\ UNCHANGED engaged
BrakeStop  == /\ speed = 1
              /\ speed' = 0
              /\ engaged' = 0

Next == Engage \/ Disengage \/ Accelerate \/ Brake \/ BrakeStop \/ UNCHANGED vars

Spec == Init /\ [][Next]_vars

TypeOK == /\ speed \in 0..Max
          /\ engaged \in {0, 1}
SafetyBounded == speed >= 0 /\ speed <= Max
SafetyValid == (engaged = 1) => (speed > 0)
====
"""))

# ---------------------------------------------------------------------------
# 15. Email inbox (receive/read/delete, bounded capacity)
# ---------------------------------------------------------------------------
SPECS.append(("manual3_email_inbox",
"Model an email inbox with receive, read, and delete operations. The inbox has a bounded capacity. Track unread and total messages.",
r"""---- MODULE EmailInbox ----
EXTENDS Integers
CONSTANT Max
VARIABLES unread, read

vars == <<unread, read>>

Init == unread = 0 /\ read = 0

Receive    == /\ unread + read < Max
              /\ unread' = unread + 1
              /\ UNCHANGED read

ReadMsg    == /\ unread > 0
              /\ unread' = unread - 1
              /\ read' = read + 1

DeleteRead == /\ read > 0
              /\ read' = read - 1
              /\ UNCHANGED unread

Next == Receive \/ ReadMsg \/ DeleteRead \/ UNCHANGED vars

Spec == Init /\ [][Next]_vars

TypeOK == /\ unread \in 0..Max
          /\ read \in 0..Max
SafetyBounded == unread + read <= Max
SafetyValid == unread >= 0 /\ read >= 0
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
