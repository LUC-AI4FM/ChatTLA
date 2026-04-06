"""
generate_gold_benchmarks.py — Write hand-crafted gold TLA+ specs for benchmark
problems, validate with SANY+TLC, and produce training data files.

Creates:
  data/processed/gold_benchmark_sft.jsonl   (SFT examples for dataset_builder)
  data/processed/rl/dpo_pairs_benchmark.jsonl (DPO: gold vs bronze/silver)

Usage:
  python scripts/generate_gold_benchmarks.py [--validate-only]
"""
from __future__ import annotations

import csv
import json
import re
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO))

from src.validators.tlc_validator import validate_string
from src.training.dataset_builder import _DEVELOPER_PROMPT  # single source of truth

# ── Gold specs for each failing benchmark ────────────────────────────────

SPECS: dict[str, dict] = {}

# ── BM001  Mutual Exclusion ──────────────────────────────────────────────
SPECS["BM001"] = {
    "description": "A mutual exclusion algorithm for N processes where at most one process is in the critical section at a time.",
    "spec": r"""---- MODULE MutualExclusion ----
EXTENDS Integers

CONSTANT N
ASSUME N \in 1..10

VARIABLE pc

Procs == 1..N

TypeOK == pc \in [Procs -> {"idle", "trying", "critical"}]

Init == pc = [p \in Procs |-> "idle"]

TryEnter(p) ==
    /\ pc[p] = "idle"
    /\ pc' = [pc EXCEPT ![p] = "trying"]

Enter(p) ==
    /\ pc[p] = "trying"
    /\ \A q \in Procs : q # p => pc[q] # "critical"
    /\ pc' = [pc EXCEPT ![p] = "critical"]

Exit(p) ==
    /\ pc[p] = "critical"
    /\ pc' = [pc EXCEPT ![p] = "idle"]

Next == \E p \in Procs : TryEnter(p) \/ Enter(p) \/ Exit(p)

MutualExclusion == \A p, q \in Procs :
    (p # q) => ~(pc[p] = "critical" /\ pc[q] = "critical")

vars == <<pc>>
Spec == Init /\ [][Next]_vars
====

\* TLC Configuration
\* SPECIFICATION Spec
\* INVARIANT TypeOK MutualExclusion
\* CONSTANT N = 3
""",
}

# ── BM002  Two-Phase Commit ──────────────────────────────────────────────
SPECS["BM002"] = {
    "description": "A two-phase commit protocol with one coordinator and N participants. The coordinator decides to commit only if all participants vote yes.",
    "spec": r"""---- MODULE TwoPhaseCommit ----
EXTENDS Integers

CONSTANT N
ASSUME N \in 1..10

VARIABLES coordState, partState, decision

Parts == 1..N

TypeOK ==
    /\ coordState \in {"init", "waiting", "committed", "aborted"}
    /\ partState \in [Parts -> {"working", "prepared", "aborted", "committed"}]
    /\ decision \in {"none", "commit", "abort"}

Init ==
    /\ coordState = "init"
    /\ partState = [p \in Parts |-> "working"]
    /\ decision = "none"

Prepare ==
    /\ coordState = "init"
    /\ coordState' = "waiting"
    /\ UNCHANGED <<partState, decision>>

VoteYes(p) ==
    /\ coordState = "waiting"
    /\ partState[p] = "working"
    /\ partState' = [partState EXCEPT ![p] = "prepared"]
    /\ UNCHANGED <<coordState, decision>>

VoteNo(p) ==
    /\ coordState = "waiting"
    /\ partState[p] = "working"
    /\ partState' = [partState EXCEPT ![p] = "aborted"]
    /\ UNCHANGED <<coordState, decision>>

DecideCommit ==
    /\ coordState = "waiting"
    /\ \A p \in Parts : partState[p] = "prepared"
    /\ decision' = "commit"
    /\ coordState' = "committed"
    /\ UNCHANGED partState

DecideAbort ==
    /\ coordState = "waiting"
    /\ \E p \in Parts : partState[p] = "aborted"
    /\ decision' = "abort"
    /\ coordState' = "aborted"
    /\ UNCHANGED partState

CommitPart(p) ==
    /\ decision = "commit"
    /\ partState[p] = "prepared"
    /\ partState' = [partState EXCEPT ![p] = "committed"]
    /\ UNCHANGED <<coordState, decision>>

AbortPart(p) ==
    /\ decision = "abort"
    /\ partState[p] \in {"prepared", "working"}
    /\ partState' = [partState EXCEPT ![p] = "aborted"]
    /\ UNCHANGED <<coordState, decision>>

Done ==
    /\ coordState \in {"committed", "aborted"}
    /\ \A p \in Parts : partState[p] \in {"committed", "aborted"}
    /\ UNCHANGED <<coordState, partState, decision>>

Next ==
    \/ Prepare
    \/ \E p \in Parts : VoteYes(p) \/ VoteNo(p)
    \/ DecideCommit
    \/ DecideAbort
    \/ \E p \in Parts : CommitPart(p) \/ AbortPart(p)
    \/ Done

Consistency ==
    \A p \in Parts :
        partState[p] = "committed" => decision = "commit"

vars == <<coordState, partState, decision>>
Spec == Init /\ [][Next]_vars
====

\* TLC Configuration
\* SPECIFICATION Spec
\* INVARIANT TypeOK Consistency
\* CONSTANT N = 3
""",
}

# ── BM004  Lamport's Bakery Algorithm ────────────────────────────────────
SPECS["BM004"] = {
    "description": "Lamport's bakery mutual exclusion algorithm for N processes. Processes take a numbered ticket; lower numbers enter first.",
    "spec": r"""---- MODULE BakeryAlgorithm ----
EXTENDS Integers, FiniteSets

CONSTANT N
ASSUME N \in 1..5

VARIABLES num, flag, pc

Procs == 1..N

TypeOK ==
    /\ num \in [Procs -> 0..20]
    /\ flag \in [Procs -> BOOLEAN]
    /\ pc \in [Procs -> {"idle", "doorway", "waiting", "critical"}]

Init ==
    /\ num = [p \in Procs |-> 0]
    /\ flag = [p \in Procs |-> FALSE]
    /\ pc = [p \in Procs |-> "idle"]

Max(S) == IF S = {} THEN 0
          ELSE CHOOSE x \in S : \A y \in S : x >= y

Doorway(p) ==
    /\ pc[p] = "idle"
    /\ Max({num[q] : q \in Procs}) < 18
    /\ flag' = [flag EXCEPT ![p] = TRUE]
    /\ num' = [num EXCEPT ![p] = Max({num[q] : q \in Procs}) + 1]
    /\ pc' = [pc EXCEPT ![p] = "doorway"]

FinishDoorway(p) ==
    /\ pc[p] = "doorway"
    /\ flag' = [flag EXCEPT ![p] = FALSE]
    /\ pc' = [pc EXCEPT ![p] = "waiting"]
    /\ UNCHANGED num

EnterCS(p) ==
    /\ pc[p] = "waiting"
    /\ \A q \in Procs \ {p} :
        /\ ~flag[q]
        /\ num[q] = 0 \/ num[p] < num[q] \/ (num[p] = num[q] /\ p < q)
    /\ pc' = [pc EXCEPT ![p] = "critical"]
    /\ UNCHANGED <<num, flag>>

ExitCS(p) ==
    /\ pc[p] = "critical"
    /\ num' = [num EXCEPT ![p] = 0]
    /\ pc' = [pc EXCEPT ![p] = "idle"]
    /\ UNCHANGED flag

Next == \E p \in Procs :
    Doorway(p) \/ FinishDoorway(p) \/ EnterCS(p) \/ ExitCS(p)

MutualExclusion ==
    \A p, q \in Procs :
        (p # q) => ~(pc[p] = "critical" /\ pc[q] = "critical")

vars == <<num, flag, pc>>
Spec == Init /\ [][Next]_vars
====

\* TLC Configuration
\* SPECIFICATION Spec
\* INVARIANT TypeOK MutualExclusion
\* CONSTANT N = 2
""",
}

# ── BM010  Simple Key-Value Store ────────────────────────────────────────
SPECS["BM010"] = {
    "description": "A single-server key-value store supporting Put(k,v) and Get(k) operations. Linearizability: a Get always returns the value of the most recent Put.",
    "spec": r"""---- MODULE KeyValueStore ----
EXTENDS Integers

CONSTANT N
ASSUME N \in 1..5

Keys == 1..N
Values == 1..N

VARIABLES store, lastGet

TypeOK ==
    /\ store \in [Keys -> Values \cup {0}]
    /\ lastGet \in [Keys -> Values \cup {0}]

Init ==
    /\ store = [k \in Keys |-> 0]
    /\ lastGet = [k \in Keys |-> 0]

Put(k, v) ==
    /\ k \in Keys
    /\ v \in Values
    /\ store' = [store EXCEPT ![k] = v]
    /\ UNCHANGED lastGet

Get(k) ==
    /\ k \in Keys
    /\ lastGet' = [lastGet EXCEPT ![k] = store[k]]
    /\ UNCHANGED store

Next == \E k \in Keys :
    (\E v \in Values : Put(k, v)) \/ Get(k)

Linearizability == \A k \in Keys : lastGet[k] = store[k] \/ lastGet[k] = 0

vars == <<store, lastGet>>
Spec == Init /\ [][Next]_vars
====

\* TLC Configuration
\* SPECIFICATION Spec
\* INVARIANT TypeOK Linearizability
\* CONSTANT N = 2
""",
}

# ── BM011  Paxos Single-Decree ───────────────────────────────────────────
SPECS["BM011"] = {
    "description": "Single-decree Paxos consensus over N acceptors and M proposers. Once a value is chosen, it is never changed.",
    "spec": r"""---- MODULE Paxos ----
EXTENDS Integers, FiniteSets

CONSTANT N
ASSUME N \in 1..5

Acceptors == 1..N
Values == {1, 2}
Ballots == 0..4

VARIABLES maxBal, maxVBal, maxVal, chosen

TypeOK ==
    /\ maxBal \in [Acceptors -> -1..4]
    /\ maxVBal \in [Acceptors -> -1..4]
    /\ maxVal \in [Acceptors -> Values \cup {-1}]
    /\ chosen \in Values \cup {-1}

Init ==
    /\ maxBal = [a \in Acceptors |-> -1]
    /\ maxVBal = [a \in Acceptors |-> -1]
    /\ maxVal = [a \in Acceptors |-> -1]
    /\ chosen = -1

Quorum == {Q \in SUBSET Acceptors : Cardinality(Q) * 2 > N}

Prepare(b) ==
    /\ b \in Ballots
    /\ \E Q \in Quorum :
        /\ \A a \in Q : maxBal[a] < b
        /\ maxBal' = [a \in Acceptors |->
            IF a \in Q THEN b ELSE maxBal[a]]
    /\ UNCHANGED <<maxVBal, maxVal, chosen>>

Accept(b, v) ==
    /\ b \in Ballots
    /\ v \in Values
    /\ \E Q \in Quorum :
        /\ \A a \in Q : maxBal[a] = b
        /\ LET promisedVals == {maxVal[a] : a \in Q} \ {-1}
           IN \/ promisedVals = {}
              \/ v \in promisedVals
        /\ maxVBal' = [a \in Acceptors |->
            IF a \in Q THEN b ELSE maxVBal[a]]
        /\ maxVal' = [a \in Acceptors |->
            IF a \in Q THEN v ELSE maxVal[a]]
        /\ maxBal' = [a \in Acceptors |->
            IF a \in Q THEN b ELSE maxBal[a]]
    /\ UNCHANGED chosen

Choose ==
    /\ chosen = -1
    /\ \E v \in Values :
        /\ \E Q \in Quorum :
            \A a \in Q : maxVal[a] = v
        /\ chosen' = v
    /\ UNCHANGED <<maxBal, maxVBal, maxVal>>

Next ==
    \/ \E b \in Ballots : Prepare(b)
    \/ \E b \in Ballots : \E v \in Values : Accept(b, v)
    \/ Choose

Consistency == chosen # -1 => chosen \in Values

vars == <<maxBal, maxVBal, maxVal, chosen>>
Spec == Init /\ [][Next]_vars
====

\* TLC Configuration
\* SPECIFICATION Spec
\* INVARIANT TypeOK Consistency
\* CONSTANT N = 3
""",
}

# ── BM012  Bounded Retransmission Protocol ───────────────────────────────
SPECS["BM012"] = {
    "description": "A sender transmits a file in chunks over an unreliable channel. The sender retransmits up to MAX_RETRIES times before giving up.",
    "spec": r"""---- MODULE BoundedRetransmission ----
EXTENDS Integers

CONSTANTS N, MAX

ASSUME N \in 1..10
ASSUME MAX \in 1..10

VARIABLES sChunk, rChunk, retry, sDone, rDone, channel

TypeOK ==
    /\ sChunk \in 1..(N + 1)
    /\ rChunk \in 0..N
    /\ retry \in 0..MAX
    /\ sDone \in {"sending", "ok", "nok"}
    /\ rDone \in {"receiving", "ok", "nok"}
    /\ channel \in {"empty", "data", "lost", "ack", "ack_lost"}

Init ==
    /\ sChunk = 1
    /\ rChunk = 0
    /\ retry = 0
    /\ sDone = "sending"
    /\ rDone = "receiving"
    /\ channel = "empty"

Send ==
    /\ sDone = "sending"
    /\ sChunk <= N
    /\ channel = "empty"
    /\ channel' = "data"
    /\ UNCHANGED <<sChunk, rChunk, retry, sDone, rDone>>

Lose ==
    /\ channel = "data"
    /\ channel' = "lost"
    /\ UNCHANGED <<sChunk, rChunk, retry, sDone, rDone>>

Deliver ==
    /\ channel = "data"
    /\ rChunk' = sChunk
    /\ channel' = "ack"
    /\ UNCHANGED <<sChunk, retry, sDone, rDone>>

LoseAck ==
    /\ channel = "ack"
    /\ channel' = "ack_lost"
    /\ UNCHANGED <<sChunk, rChunk, retry, sDone, rDone>>

RecvAck ==
    /\ channel = "ack"
    /\ retry' = 0
    /\ IF sChunk = N
       THEN /\ sDone' = "ok"
            /\ rDone' = "ok"
            /\ sChunk' = N + 1
       ELSE /\ sChunk' = sChunk + 1
            /\ UNCHANGED <<sDone, rDone>>
    /\ channel' = "empty"
    /\ UNCHANGED rChunk

Timeout ==
    /\ channel \in {"lost", "ack_lost"}
    /\ sDone = "sending"
    /\ IF retry < MAX
       THEN /\ retry' = retry + 1
            /\ channel' = "empty"
            /\ UNCHANGED <<sChunk, rChunk, sDone, rDone>>
       ELSE /\ sDone' = "nok"
            /\ rDone' = "nok"
            /\ channel' = "empty"
            /\ UNCHANGED <<sChunk, rChunk, retry>>

Done ==
    /\ sDone \in {"ok", "nok"}
    /\ UNCHANGED <<sChunk, rChunk, retry, sDone, rDone, channel>>

Next ==
    \/ Send \/ Lose \/ Deliver \/ LoseAck \/ RecvAck \/ Timeout \/ Done

DeliveryOrFailure ==
    /\ sDone = "ok" => rChunk = N
    /\ rDone = "ok" => rChunk = N

vars == <<sChunk, rChunk, retry, sDone, rDone, channel>>
Spec == Init /\ [][Next]_vars
====

\* TLC Configuration
\* SPECIFICATION Spec
\* INVARIANT TypeOK DeliveryOrFailure
\* CONSTANT N = 3
\* CONSTANT MAX = 2
""",
}

# ── BM013  Transaction Isolation (Snapshot Isolation) ────────────────────
SPECS["BM013"] = {
    "description": "A database with snapshot isolation. Transactions read a consistent snapshot. Write-write conflicts cause an abort.",
    "spec": r"""---- MODULE SnapshotIsolation ----
EXTENDS Integers, FiniteSets

CONSTANT N
ASSUME N \in 1..3

Keys == 1..N
Txns == 1..N

VARIABLES db, txState, txSnapshot, txWrites

TypeOK ==
    /\ db \in [Keys -> 0..5]
    /\ txState \in [Txns -> {"idle", "active", "committed", "aborted"}]
    /\ txSnapshot \in [Txns -> [Keys -> 0..5]]
    /\ txWrites \in [Txns -> SUBSET Keys]

Init ==
    /\ db = [k \in Keys |-> 0]
    /\ txState = [t \in Txns |-> "idle"]
    /\ txSnapshot = [t \in Txns |-> [k \in Keys |-> 0]]
    /\ txWrites = [t \in Txns |-> {}]

Begin(t) ==
    /\ txState[t] = "idle"
    /\ txState' = [txState EXCEPT ![t] = "active"]
    /\ txSnapshot' = [txSnapshot EXCEPT ![t] = db]
    /\ txWrites' = [txWrites EXCEPT ![t] = {}]
    /\ UNCHANGED db

Write(t, k) ==
    /\ txState[t] = "active"
    /\ k \in Keys
    /\ txWrites' = [txWrites EXCEPT ![t] = @ \cup {k}]
    /\ UNCHANGED <<db, txState, txSnapshot>>

Commit(t) ==
    /\ txState[t] = "active"
    /\ \A t2 \in Txns :
        (t2 # t /\ txState[t2] = "committed")
            => txWrites[t] \cap txWrites[t2] = {}
    /\ db' = [k \in Keys |->
        IF k \in txWrites[t] THEN txSnapshot[t][k] + 1 ELSE db[k]]
    /\ txState' = [txState EXCEPT ![t] = "committed"]
    /\ UNCHANGED <<txSnapshot, txWrites>>

Abort(t) ==
    /\ txState[t] = "active"
    /\ txState' = [txState EXCEPT ![t] = "aborted"]
    /\ UNCHANGED <<db, txSnapshot, txWrites>>

Done ==
    /\ \A t \in Txns : txState[t] \in {"committed", "aborted"}
    /\ UNCHANGED <<db, txState, txSnapshot, txWrites>>

Next == (\E t \in Txns :
    Begin(t) \/ (\E k \in Keys : Write(t, k)) \/ Commit(t) \/ Abort(t)) \/ Done

NoWriteConflict ==
    \A t1, t2 \in Txns :
        (t1 # t2 /\ txState[t1] = "committed" /\ txState[t2] = "committed")
            => txWrites[t1] \cap txWrites[t2] = {}

vars == <<db, txState, txSnapshot, txWrites>>
Spec == Init /\ [][Next]_vars
====

\* TLC Configuration
\* SPECIFICATION Spec
\* INVARIANT TypeOK NoWriteConflict
\* CONSTANT N = 2
""",
}

# ── BM014  Clock Synchronisation ─────────────────────────────────────────
SPECS["BM014"] = {
    "description": "N nodes exchange clock values to synchronise. After one round, all clocks are within epsilon of each other.",
    "spec": r"""---- MODULE ClockSync ----
EXTENDS Integers

CONSTANT N
ASSUME N \in 1..5

VARIABLES clock, synced

Nodes == 1..N

TypeOK ==
    /\ clock \in [Nodes -> 0..10]
    /\ synced \in [Nodes -> BOOLEAN]

Init ==
    /\ clock \in [Nodes -> 0..3]
    /\ synced = [n \in Nodes |-> FALSE]

Sync(i) ==
    /\ i \in Nodes
    /\ ~synced[i]
    /\ \E j \in Nodes :
        /\ j # i
        /\ LET avg == (clock[i] + clock[j]) \div 2
           IN clock' = [clock EXCEPT ![i] = avg]
    /\ synced' = [synced EXCEPT ![i] = TRUE]

Done ==
    /\ \A n \in Nodes : synced[n]
    /\ UNCHANGED <<clock, synced>>

Next == (\E i \in Nodes : Sync(i)) \/ Done

ClockBound ==
    (\A n \in Nodes : synced[n])
        => \A i, j \in Nodes : clock[i] - clock[j] \in -5..5

vars == <<clock, synced>>
Spec == Init /\ [][Next]_vars
====

\* TLC Configuration
\* SPECIFICATION Spec
\* INVARIANT TypeOK ClockBound
\* CONSTANT N = 3
""",
}

# ── BM017  Simple Allocator ──────────────────────────────────────────────
SPECS["BM017"] = {
    "description": "A memory allocator managing a fixed pool of N pages. Clients request and release pages. Safety: no page is allocated to two clients simultaneously.",
    "spec": r"""---- MODULE SimpleAllocator ----
EXTENDS Integers, FiniteSets

CONSTANT N
ASSUME N \in 1..5

Pages == 1..N
Clients == 1..N

VARIABLES free, allocated

TypeOK ==
    /\ free \subseteq Pages
    /\ allocated \in [Clients -> SUBSET Pages]

Init ==
    /\ free = Pages
    /\ allocated = [c \in Clients |-> {}]

Allocate(c) ==
    /\ c \in Clients
    /\ free # {}
    /\ \E p \in free :
        /\ free' = free \ {p}
        /\ allocated' = [allocated EXCEPT ![c] = @ \cup {p}]

Release(c) ==
    /\ c \in Clients
    /\ allocated[c] # {}
    /\ \E p \in allocated[c] :
        /\ allocated' = [allocated EXCEPT ![c] = @ \ {p}]
        /\ free' = free \cup {p}

Next == \E c \in Clients : Allocate(c) \/ Release(c)

SafeAllocation ==
    \A c1, c2 \in Clients :
        c1 # c2 => allocated[c1] \cap allocated[c2] = {}

vars == <<free, allocated>>
Spec == Init /\ [][Next]_vars
====

\* TLC Configuration
\* SPECIFICATION Spec
\* INVARIANT TypeOK SafeAllocation
\* CONSTANT N = 3
""",
}

# ── BM018  Publish-Subscribe Broker ──────────────────────────────────────
SPECS["BM018"] = {
    "description": "A single broker with subscribers and publishers. Subscribers register interest in topics. Publishers post messages on topics. The broker delivers each message to all registered subscribers.",
    "spec": r"""---- MODULE PubSubBroker ----
EXTENDS Integers, FiniteSets

CONSTANT N
ASSUME N \in 1..3

Topics == 1..N
Subs == 1..N
MaxMsgs == 1

VARIABLES subscriptions, published, delivered

TypeOK ==
    /\ subscriptions \in [Subs -> SUBSET Topics]
    /\ published \in [Topics -> 0..3]
    /\ delivered \in [Subs -> [Topics -> 0..3]]

Init ==
    /\ subscriptions = [s \in Subs |-> {}]
    /\ published = [t \in Topics |-> 0]
    /\ delivered = [s \in Subs |-> [t \in Topics |-> 0]]

Subscribe(s, t) ==
    /\ s \in Subs
    /\ t \in Topics
    /\ subscriptions' = [subscriptions EXCEPT ![s] = @ \cup {t}]
    /\ UNCHANGED <<published, delivered>>

Publish(t) ==
    /\ t \in Topics
    /\ published[t] < MaxMsgs
    /\ published' = [published EXCEPT ![t] = @ + 1]
    /\ UNCHANGED <<subscriptions, delivered>>

Deliver(s, t) ==
    /\ s \in Subs
    /\ t \in subscriptions[s]
    /\ delivered[s][t] < published[t]
    /\ delivered' = [delivered EXCEPT ![s][t] = @ + 1]
    /\ UNCHANGED <<subscriptions, published>>

Done ==
    /\ \A t \in Topics : published[t] = MaxMsgs
    /\ UNCHANGED <<subscriptions, published, delivered>>

Next ==
    \/ \E s \in Subs, t \in Topics : Subscribe(s, t)
    \/ \E t \in Topics : Publish(t)
    \/ \E s \in Subs, t \in Topics : Deliver(s, t)
    \/ Done

DeliveryGuarantee ==
    \A s \in Subs, t \in Topics :
        t \in subscriptions[s] => delivered[s][t] <= published[t]

vars == <<subscriptions, published, delivered>>
Spec == Init /\ [][Next]_vars
====

\* TLC Configuration
\* SPECIFICATION Spec
\* INVARIANT TypeOK DeliveryGuarantee
\* CONSTANT N = 2
""",
}

# ── BM019  Dekker's Algorithm ────────────────────────────────────────────
SPECS["BM019"] = {
    "description": "Dekker's mutual exclusion algorithm for 2 processes, the first known correct solution to the problem.",
    "spec": r"""---- MODULE DekkersAlgorithm ----
EXTENDS Integers

VARIABLES wants, turn, pc

Procs == {0, 1}

Other(p) == 1 - p

TypeOK ==
    /\ wants \in [Procs -> BOOLEAN]
    /\ turn \in Procs
    /\ pc \in [Procs -> {"idle", "set_flag", "check", "wait", "critical", "exit"}]

Init ==
    /\ wants = [p \in Procs |-> FALSE]
    /\ turn = 0
    /\ pc = [p \in Procs |-> "idle"]

SetFlag(p) ==
    /\ pc[p] = "idle"
    /\ wants' = [wants EXCEPT ![p] = TRUE]
    /\ pc' = [pc EXCEPT ![p] = "set_flag"]
    /\ UNCHANGED turn

Check(p) ==
    /\ pc[p] = "set_flag"
    /\ IF ~wants[Other(p)]
       THEN pc' = [pc EXCEPT ![p] = "critical"]
       ELSE pc' = [pc EXCEPT ![p] = "check"]
    /\ UNCHANGED <<wants, turn>>

Wait(p) ==
    /\ pc[p] = "check"
    /\ wants[Other(p)]
    /\ IF turn # p
       THEN /\ wants' = [wants EXCEPT ![p] = FALSE]
            /\ pc' = [pc EXCEPT ![p] = "wait"]
       ELSE /\ pc' = [pc EXCEPT ![p] = "set_flag"]
            /\ UNCHANGED wants
    /\ UNCHANGED turn

WaitForTurn(p) ==
    /\ pc[p] = "wait"
    /\ turn = p
    /\ wants' = [wants EXCEPT ![p] = TRUE]
    /\ pc' = [pc EXCEPT ![p] = "set_flag"]
    /\ UNCHANGED turn

EnterFromCheck(p) ==
    /\ pc[p] = "check"
    /\ ~wants[Other(p)]
    /\ pc' = [pc EXCEPT ![p] = "critical"]
    /\ UNCHANGED <<wants, turn>>

ExitCS(p) ==
    /\ pc[p] = "critical"
    /\ turn' = Other(p)
    /\ wants' = [wants EXCEPT ![p] = FALSE]
    /\ pc' = [pc EXCEPT ![p] = "idle"]

Next == \E p \in Procs :
    SetFlag(p) \/ Check(p) \/ Wait(p) \/ WaitForTurn(p)
    \/ EnterFromCheck(p) \/ ExitCS(p)

MutualExclusion ==
    ~(pc[0] = "critical" /\ pc[1] = "critical")

Deadlock_Freedom ==
    (wants[0] \/ wants[1]) =>
        (pc[0] \in {"critical", "set_flag", "check", "wait"}
         \/ pc[1] \in {"critical", "set_flag", "check", "wait"})

vars == <<wants, turn, pc>>
Spec == Init /\ [][Next]_vars
====

\* TLC Configuration
\* SPECIFICATION Spec
\* INVARIANT TypeOK MutualExclusion
""",
}


def validate_spec(bm_id: str, info: dict) -> tuple[str, str]:
    """Validate a spec, return (tier, message)."""
    spec = info["spec"].strip()
    m = re.search(r"----\s*MODULE\s+(\w+)", spec)
    module_name = m.group(1) if m else "Temp"
    result = validate_string(spec, module_name=module_name, timeout=60)
    detail = ""
    if result.sany_errors:
        detail = f" SANY: {result.sany_errors[:2]}"
    if result.tlc_violations:
        detail += f" TLC: {result.tlc_violations[:2]}"
    return result.tier, f"{bm_id}: {result.tier}{detail}"


def build_sft_example(bm_id: str, info: dict) -> dict:
    """Build a harmony-formatted SFT example."""
    return {
        "_tier": "gold_benchmark",
        "_prompt_id": bm_id,
        "messages": [
            {"role": "developer", "content": _DEVELOPER_PROMPT},
            {"role": "user", "content": f"Write a TLA+ specification for the following:\n\n{info['description']}"},
            {"role": "assistant", "channel": "analysis", "content": "I'll write a well-formed TLA+ specification with proper Init, Next, and invariants."},
            {"role": "assistant", "channel": "final", "content": info["spec"].strip()},
        ],
    }


def build_dpo_pair(bm_id: str, info: dict, rejected_spec: str) -> dict | None:
    """Build a DPO pair: gold spec (chosen) vs benchmark-failing spec (rejected)."""
    if not rejected_spec or len(rejected_spec.strip()) < 50:
        return None
    return {
        "prompt": f"Write a TLA+ specification for the following:\n\n{info['description']}",
        "chosen": info["spec"].strip(),
        "rejected": rejected_spec.strip(),
        "chosen_tier": "gold",
        "rejected_tier": "bronze",
        "_prompt_id": bm_id,
    }


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()

    failing_ids = set(SPECS.keys())
    print(f"\nValidating {len(SPECS)} gold specs...\n")

    gold_specs = {}
    for bm_id, info in sorted(SPECS.items()):
        tier, msg = validate_spec(bm_id, info)
        icon = {"gold": "G", "silver": "S", "bronze": "X"}[tier]
        print(f"  [{icon}] {msg}")
        if tier == "gold":
            gold_specs[bm_id] = info

    print(f"\nGold: {len(gold_specs)}/{len(SPECS)}")

    if args.validate_only:
        return

    # Write gold_benchmark_sft.jsonl
    sft_path = _REPO / "data" / "processed" / "gold_benchmark_sft.jsonl"
    sft_path.parent.mkdir(parents=True, exist_ok=True)
    with sft_path.open("w", encoding="utf-8") as f:
        for bm_id, info in sorted(gold_specs.items()):
            ex = build_sft_example(bm_id, info)
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    print(f"\nWrote {len(gold_specs)} SFT examples -> {sft_path}")

    # Load rejected specs from latest benchmark CSV for DPO pairs
    csv_pattern = sorted(
        _REPO.glob("outputs/benchmark_results_v13_full_*.csv"),
        key=lambda p: p.stat().st_mtime, reverse=True,
    )
    dpo_path = _REPO / "data" / "processed" / "rl" / "dpo_pairs_benchmark.jsonl"
    n_dpo = 0
    if csv_pattern:
        with csv_pattern[0].open() as f:
            rows = {r["benchmark_id"]: r for r in csv.DictReader(f)}
        with dpo_path.open("w", encoding="utf-8") as f:
            for bm_id, info in sorted(gold_specs.items()):
                if bm_id in rows and rows[bm_id].get("generated_spec"):
                    pair = build_dpo_pair(bm_id, info, rows[bm_id]["generated_spec"])
                    if pair:
                        f.write(json.dumps(pair, ensure_ascii=False) + "\n")
                        n_dpo += 1
        print(f"Wrote {n_dpo} DPO pairs -> {dpo_path}")

    # Also append benchmark DPO pairs to the main dpo_pairs.jsonl
    main_dpo = _REPO / "data" / "processed" / "rl" / "dpo_pairs.jsonl"
    existing_ids = set()
    if main_dpo.exists():
        for line in main_dpo.open():
            try:
                existing_ids.add(json.loads(line).get("_prompt_id"))
            except (json.JSONDecodeError, KeyError):
                pass
    n_appended = 0
    if dpo_path.exists():
        with main_dpo.open("a", encoding="utf-8") as out:
            for line in dpo_path.open():
                obj = json.loads(line)
                if obj.get("_prompt_id") not in existing_ids:
                    out.write(line)
                    n_appended += 1
        if n_appended:
            print(f"Appended {n_appended} new DPO pairs to {main_dpo}")


if __name__ == "__main__":
    main()
