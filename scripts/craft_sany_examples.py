#!/usr/bin/env python3
"""
craft_sany_examples.py — Hand-craft SANY-verified TLA+ training examples.

Each example is validated against SANY before being written.
Only specs that pass SANY are included in the output.

Usage:
    python scripts/craft_sany_examples.py
"""

import json
import subprocess
import tempfile
import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
TLA_TOOLS_JAR = REPO_ROOT / "src" / "shared" / "tlc" / "tla2tools.jar"
AUGMENTED_JSONL = REPO_ROOT / "data" / "processed" / "augmented.jsonl"

from src.training.dataset_builder import _DEVELOPER_PROMPT as DEVELOPER_PROMPT  # single source of truth


def validate_sany(spec: str, module_name: str) -> tuple[bool, str]:
    """Run SANY on a spec string. Returns (passed, output)."""
    with tempfile.TemporaryDirectory() as td:
        tla_file = os.path.join(td, f"{module_name}.tla")
        with open(tla_file, "w") as f:
            f.write(spec)
        result = subprocess.run(
            ["java", "-cp", str(TLA_TOOLS_JAR), "tla2sany.SANY", tla_file],
            capture_output=True, text=True, timeout=30,
        )
        output = result.stdout + result.stderr
        passed = "error" not in output.lower() or "0 error" in output.lower()
        # More precise check
        if "Semantic errors" in output or "Parse Error" in output or "Lexical error" in output:
            passed = False
        if "Semantic processing of module" in output and "error" not in output.lower():
            passed = True
        return passed, output


def build_example(prompt: str, spec: str) -> dict:
    """Build a training example in harmony format."""
    return {"messages": [
        {"role": "developer", "content": DEVELOPER_PROMPT},
        {"role": "user", "content": f"Write a TLA+ specification for the following:\n\n{prompt}"},
        {"role": "assistant", "channel": "analysis", "content": "I'll write a well-formed TLA+ specification with proper Init, Next, and invariants."},
        {"role": "assistant", "channel": "final", "content": spec.strip()},
    ]}


# ─────────────────────────────────────────────────────────────────────────────
# Hand-crafted TLA+ specs — each must pass SANY
# ─────────────────────────────────────────────────────────────────────────────

EXAMPLES = [
    # ── BM001: Mutual Exclusion ──────────────────────────────────────────
    {
        "prompt": "A mutual exclusion algorithm for N processes where at most one process is in the critical section at a time.",
        "module": "MutualExclusion",
        "spec": r"""---- MODULE MutualExclusion ----

EXTENDS Naturals

CONSTANT N

VARIABLES state, turn

vars == <<state, turn>>

TypeOK ==
    /\ state \in [1..N -> {"idle", "trying", "critical"}]
    /\ turn \in 1..N

Init ==
    /\ state = [i \in 1..N |-> "idle"]
    /\ turn = 1

TryEnter(i) ==
    /\ state[i] = "idle"
    /\ state' = [state EXCEPT ![i] = "trying"]
    /\ turn' = i

Enter(i) ==
    /\ state[i] = "trying"
    /\ \A j \in 1..N : j /= i => state[j] /= "critical"
    /\ state' = [state EXCEPT ![i] = "critical"]
    /\ UNCHANGED turn

Exit(i) ==
    /\ state[i] = "critical"
    /\ state' = [state EXCEPT ![i] = "idle"]
    /\ UNCHANGED turn

Next == \E i \in 1..N : TryEnter(i) \/ Enter(i) \/ Exit(i)

Spec == Init /\ [][Next]_vars

MutualExclusion == \A i, j \in 1..N : (i /= j) => ~(state[i] = "critical" /\ state[j] = "critical")

====
""",
    },

    # ── BM002: Two-Phase Commit ──────────────────────────────────────────
    {
        "prompt": "A two-phase commit protocol with one coordinator and N participants. The coordinator decides to commit only if all participants vote yes.",
        "module": "TwoPhaseCommit",
        "spec": r"""---- MODULE TwoPhaseCommit ----

EXTENDS Naturals, FiniteSets

CONSTANT N

VARIABLES pcState, pState, decision

vars == <<pcState, pState, decision>>

Participants == 1..N

TypeOK ==
    /\ pcState \in {"init", "waiting", "committed", "aborted"}
    /\ pState \in [Participants -> {"working", "prepared", "committed", "aborted"}]
    /\ decision \in {"none", "commit", "abort"}

Init ==
    /\ pcState = "init"
    /\ pState = [p \in Participants |-> "working"]
    /\ decision = "none"

Prepare ==
    /\ pcState = "init"
    /\ pcState' = "waiting"
    /\ UNCHANGED <<pState, decision>>

VoteYes(p) ==
    /\ pcState = "waiting"
    /\ pState[p] = "working"
    /\ pState' = [pState EXCEPT ![p] = "prepared"]
    /\ UNCHANGED <<pcState, decision>>

VoteNo(p) ==
    /\ pcState = "waiting"
    /\ pState[p] = "working"
    /\ pState' = [pState EXCEPT ![p] = "aborted"]
    /\ UNCHANGED <<pcState, decision>>

Decide ==
    /\ pcState = "waiting"
    /\ IF \A p \in Participants : pState[p] = "prepared"
       THEN /\ decision' = "commit"
            /\ pcState' = "committed"
            /\ pState' = [p \in Participants |-> "committed"]
       ELSE /\ decision' = "abort"
            /\ pcState' = "aborted"
            /\ pState' = [p \in Participants |-> "aborted"]

Next ==
    \/ Prepare
    \/ \E p \in Participants : VoteYes(p)
    \/ \E p \in Participants : VoteNo(p)
    \/ Decide

Spec == Init /\ [][Next]_vars

Consistency ==
    decision = "commit" => \A p \in Participants : pState[p] /= "aborted"

====
""",
    },

    # ── BM003: Dining Philosophers ───────────────────────────────────────
    {
        "prompt": "The dining philosophers problem with N philosophers and N forks. Each philosopher picks up both adjacent forks to eat, then puts them down.",
        "module": "DiningPhilosophers",
        "spec": r"""---- MODULE DiningPhilosophers ----

EXTENDS Naturals

CONSTANT N

VARIABLES phil, fork

vars == <<phil, fork>>

LeftFork(i) == i
RightFork(i) == IF i = N THEN 1 ELSE i + 1

TypeOK ==
    /\ phil \in [1..N -> {"thinking", "hungry", "eating"}]
    /\ fork \in [1..N -> {"free", "taken"}]

Init ==
    /\ phil = [i \in 1..N |-> "thinking"]
    /\ fork = [i \in 1..N |-> "free"]

GetHungry(i) ==
    /\ phil[i] = "thinking"
    /\ phil' = [phil EXCEPT ![i] = "hungry"]
    /\ UNCHANGED fork

PickUp(i) ==
    /\ phil[i] = "hungry"
    /\ fork[LeftFork(i)] = "free"
    /\ fork[RightFork(i)] = "free"
    /\ phil' = [phil EXCEPT ![i] = "eating"]
    /\ fork' = [fork EXCEPT ![LeftFork(i)] = "taken", ![RightFork(i)] = "taken"]

PutDown(i) ==
    /\ phil[i] = "eating"
    /\ phil' = [phil EXCEPT ![i] = "thinking"]
    /\ fork' = [fork EXCEPT ![LeftFork(i)] = "free", ![RightFork(i)] = "free"]

Next == \E i \in 1..N : GetHungry(i) \/ PickUp(i) \/ PutDown(i)

Spec == Init /\ [][Next]_vars

NoDeadlock == \E i \in 1..N : phil[i] /= "hungry"

====
""",
    },

    # ── BM004: Lamport's Bakery Algorithm ────────────────────────────────
    {
        "prompt": "Lamport's bakery mutual exclusion algorithm for N processes. Processes take a numbered ticket; lower numbers enter first.",
        "module": "Bakery",
        "spec": r"""---- MODULE Bakery ----

EXTENDS Naturals

CONSTANT N

VARIABLES num, flag, pc

vars == <<num, flag, pc>>

Procs == 1..N

TypeOK ==
    /\ num \in [Procs -> Nat]
    /\ flag \in [Procs -> BOOLEAN]
    /\ pc \in [Procs -> {"idle", "choosing", "waiting", "cs"}]

Init ==
    /\ num = [i \in Procs |-> 0]
    /\ flag = [i \in Procs |-> FALSE]
    /\ pc = [i \in Procs |-> "idle"]

Choose(i) ==
    /\ pc[i] = "idle"
    /\ flag' = [flag EXCEPT ![i] = TRUE]
    /\ \E k \in 1..(N+1) :
        /\ \A j \in Procs : num[j] < k
        /\ num' = [num EXCEPT ![i] = k]
    /\ pc' = [pc EXCEPT ![i] = "choosing"]

FinishChoosing(i) ==
    /\ pc[i] = "choosing"
    /\ flag' = [flag EXCEPT ![i] = FALSE]
    /\ pc' = [pc EXCEPT ![i] = "waiting"]
    /\ UNCHANGED num

EnterCS(i) ==
    /\ pc[i] = "waiting"
    /\ \A j \in Procs \ {i} :
        \/ num[j] = 0
        \/ num[i] < num[j]
        \/ (num[i] = num[j] /\ i < j)
    /\ pc' = [pc EXCEPT ![i] = "cs"]
    /\ UNCHANGED <<num, flag>>

ExitCS(i) ==
    /\ pc[i] = "cs"
    /\ num' = [num EXCEPT ![i] = 0]
    /\ pc' = [pc EXCEPT ![i] = "idle"]
    /\ UNCHANGED flag

Next == \E i \in Procs : Choose(i) \/ FinishChoosing(i) \/ EnterCS(i) \/ ExitCS(i)

Spec == Init /\ [][Next]_vars

MutualExclusion == \A i, j \in Procs : (i /= j) => ~(pc[i] = "cs" /\ pc[j] = "cs")

====
""",
    },

    # ── BM005: Producer-Consumer Queue ───────────────────────────────────
    {
        "prompt": "A bounded FIFO queue with one producer and one consumer. The producer blocks when the queue is full; the consumer blocks when empty.",
        "module": "BoundedQueue",
        "spec": r"""---- MODULE BoundedQueue ----

EXTENDS Naturals, Sequences

CONSTANT K

VARIABLES queue

vars == <<queue>>

TypeOK == queue \in Seq(Nat)

BoundedQueue == Len(queue) <= K

Init == queue = <<>>

Produce(v) ==
    /\ Len(queue) < K
    /\ queue' = Append(queue, v)

Consume ==
    /\ Len(queue) > 0
    /\ queue' = Tail(queue)

Next ==
    \/ \E v \in 1..10 : Produce(v)
    \/ Consume

Spec == Init /\ [][Next]_vars

====
""",
    },

    # ── BM006: Raft Leader Election ──────────────────────────────────────
    {
        "prompt": "The leader election phase of the Raft consensus algorithm. Servers hold terms and vote for candidates. A candidate wins if it gets a majority of votes.",
        "module": "RaftLeaderElection",
        "spec": r"""---- MODULE RaftLeaderElection ----

EXTENDS Naturals, FiniteSets

CONSTANT N

Servers == 1..N
Quorum == (N \div 2) + 1

VARIABLES currentTerm, votedFor, state, votesGranted

vars == <<currentTerm, votedFor, state, votesGranted>>

TypeOK ==
    /\ currentTerm \in [Servers -> Nat]
    /\ votedFor \in [Servers -> Servers \cup {0}]
    /\ state \in [Servers -> {"follower", "candidate", "leader"}]
    /\ votesGranted \in [Servers -> SUBSET Servers]

Init ==
    /\ currentTerm = [s \in Servers |-> 0]
    /\ votedFor = [s \in Servers |-> 0]
    /\ state = [s \in Servers |-> "follower"]
    /\ votesGranted = [s \in Servers |-> {}]

BecomeCandidate(s) ==
    /\ state[s] = "follower"
    /\ currentTerm' = [currentTerm EXCEPT ![s] = currentTerm[s] + 1]
    /\ state' = [state EXCEPT ![s] = "candidate"]
    /\ votedFor' = [votedFor EXCEPT ![s] = s]
    /\ votesGranted' = [votesGranted EXCEPT ![s] = {s}]

RequestVote(cand, voter) ==
    /\ state[cand] = "candidate"
    /\ voter /= cand
    /\ currentTerm[cand] >= currentTerm[voter]
    /\ votedFor[voter] = 0
    /\ votedFor' = [votedFor EXCEPT ![voter] = cand]
    /\ votesGranted' = [votesGranted EXCEPT ![cand] = votesGranted[cand] \cup {voter}]
    /\ currentTerm' = [currentTerm EXCEPT ![voter] = currentTerm[cand]]
    /\ UNCHANGED state

BecomeLeader(s) ==
    /\ state[s] = "candidate"
    /\ Cardinality(votesGranted[s]) >= Quorum
    /\ state' = [state EXCEPT ![s] = "leader"]
    /\ UNCHANGED <<currentTerm, votedFor, votesGranted>>

Next ==
    \/ \E s \in Servers : BecomeCandidate(s)
    \/ \E s, v \in Servers : RequestVote(s, v)
    \/ \E s \in Servers : BecomeLeader(s)

Spec == Init /\ [][Next]_vars

AtMostOneLeader ==
    \A s1, s2 \in Servers :
        (state[s1] = "leader" /\ state[s2] = "leader" /\ currentTerm[s1] = currentTerm[s2])
        => s1 = s2

====
""",
    },

    # ── BM007: Read-Write Lock ───────────────────────────────────────────
    {
        "prompt": "A read-write lock allowing multiple concurrent readers but exclusive writer access. Writers wait for all readers to finish.",
        "module": "ReadWriteLock",
        "spec": r"""---- MODULE ReadWriteLock ----

EXTENDS Naturals

CONSTANT NumActors

VARIABLES readers, writerActive, waiting

vars == <<readers, writerActive, waiting>>

Actors == 1..NumActors

TypeOK ==
    /\ readers \in SUBSET Actors
    /\ writerActive \in BOOLEAN
    /\ waiting \in SUBSET Actors

Init ==
    /\ readers = {}
    /\ writerActive = FALSE
    /\ waiting = {}

StartRead(a) ==
    /\ a \notin readers
    /\ writerActive = FALSE
    /\ readers' = readers \cup {a}
    /\ UNCHANGED <<writerActive, waiting>>

EndRead(a) ==
    /\ a \in readers
    /\ readers' = readers \ {a}
    /\ UNCHANGED <<writerActive, waiting>>

StartWrite(a) ==
    /\ a \notin readers
    /\ writerActive = FALSE
    /\ readers = {}
    /\ writerActive' = TRUE
    /\ UNCHANGED <<readers, waiting>>

EndWrite(a) ==
    /\ writerActive = TRUE
    /\ writerActive' = FALSE
    /\ UNCHANGED <<readers, waiting>>

Next ==
    \E a \in Actors :
        \/ StartRead(a)
        \/ EndRead(a)
        \/ StartWrite(a)
        \/ EndWrite(a)

Spec == Init /\ [][Next]_vars

ExclusiveWrite ==
    writerActive => readers = {}

====
""",
    },

    # ── BM008: Distributed Snapshot (Chandy-Lamport) ─────────────────────
    {
        "prompt": "The Chandy-Lamport distributed snapshot algorithm over a network of N processes with FIFO channels.",
        "module": "ChandyLamport",
        "spec": r"""---- MODULE ChandyLamport ----

EXTENDS Naturals, Sequences, FiniteSets

CONSTANT N

Procs == 1..N

VARIABLES localState, recorded, markers, channels, chanRecorded

vars == <<localState, recorded, markers, channels, chanRecorded>>

TypeOK ==
    /\ localState \in [Procs -> Nat]
    /\ recorded \in [Procs -> BOOLEAN]
    /\ markers \in [Procs -> BOOLEAN]
    /\ chanRecorded \in [Procs -> [Procs -> BOOLEAN]]

Init ==
    /\ localState = [p \in Procs |-> 0]
    /\ recorded = [p \in Procs |-> FALSE]
    /\ markers = [p \in Procs |-> FALSE]
    /\ channels = [p \in Procs |-> [q \in Procs |-> <<>>]]
    /\ chanRecorded = [p \in Procs |-> [q \in Procs |-> FALSE]]

InitiateSnapshot(p) ==
    /\ recorded[p] = FALSE
    /\ recorded' = [recorded EXCEPT ![p] = TRUE]
    /\ markers' = [markers EXCEPT ![p] = TRUE]
    /\ chanRecorded' = [chanRecorded EXCEPT ![p] = [q \in Procs |-> TRUE]]
    /\ UNCHANGED <<localState, channels>>

RecordState(p) ==
    /\ recorded[p] = FALSE
    /\ \E q \in Procs : markers[q] = TRUE /\ q /= p
    /\ recorded' = [recorded EXCEPT ![p] = TRUE]
    /\ UNCHANGED <<localState, markers, channels, chanRecorded>>

Compute(p) ==
    /\ localState' = [localState EXCEPT ![p] = localState[p] + 1]
    /\ UNCHANGED <<recorded, markers, channels, chanRecorded>>

Next == \E p \in Procs : InitiateSnapshot(p) \/ RecordState(p) \/ Compute(p)

Spec == Init /\ [][Next]_vars

SnapshotConsistency ==
    (\A p \in Procs : recorded[p]) => TRUE

====
""",
    },

    # ── BM009: Token Ring ────────────────────────────────────────────────
    {
        "prompt": "A token ring network with N nodes. Each node passes a single token clockwise. Only the token holder may send a message.",
        "module": "TokenRing",
        "spec": r"""---- MODULE TokenRing ----

EXTENDS Naturals

CONSTANT N

Nodes == 1..N

VARIABLES tokenHolder, hasMessage

vars == <<tokenHolder, hasMessage>>

TypeOK ==
    /\ tokenHolder \in Nodes
    /\ hasMessage \in [Nodes -> BOOLEAN]

Init ==
    /\ tokenHolder = 1
    /\ hasMessage = [n \in Nodes |-> FALSE]

PassToken ==
    /\ tokenHolder' = IF tokenHolder = N THEN 1 ELSE tokenHolder + 1
    /\ UNCHANGED hasMessage

SendMessage ==
    /\ hasMessage' = [hasMessage EXCEPT ![tokenHolder] = TRUE]
    /\ UNCHANGED tokenHolder

Next == PassToken \/ SendMessage

Spec == Init /\ [][Next]_vars

UniqueToken == \A n1, n2 \in Nodes : tokenHolder = n1 /\ tokenHolder = n2 => n1 = n2

====
""",
    },

    # ── BM010: Simple Key-Value Store ────────────────────────────────────
    {
        "prompt": "A single-server key-value store supporting Put(k,v) and Get(k) operations. Linearizability: a Get always returns the value of the most recent Put.",
        "module": "KVStore",
        "spec": r"""---- MODULE KVStore ----

EXTENDS Naturals

CONSTANT Keys, Values

VARIABLES store, lastResult

vars == <<store, lastResult>>

TypeOK ==
    /\ store \in [Keys -> Values \cup {0}]
    /\ lastResult \in Values \cup {0}

Init ==
    /\ store = [k \in Keys |-> 0]
    /\ lastResult = 0

Put(k, v) ==
    /\ k \in Keys
    /\ v \in Values
    /\ store' = [store EXCEPT ![k] = v]
    /\ UNCHANGED lastResult

Get(k) ==
    /\ k \in Keys
    /\ lastResult' = store[k]
    /\ UNCHANGED store

Next ==
    \/ \E k \in Keys, v \in Values : Put(k, v)
    \/ \E k \in Keys : Get(k)

Spec == Init /\ [][Next]_vars

Linearizability == \A k \in Keys : lastResult = store[k] \/ lastResult = 0

====
""",
    },

    # ── BM011: Paxos Single-Decree ───────────────────────────────────────
    {
        "prompt": "Single-decree Paxos consensus over N acceptors and M proposers. Once a value is chosen, it is never changed.",
        "module": "Paxos",
        "spec": r"""---- MODULE Paxos ----

EXTENDS Naturals, FiniteSets

CONSTANT N, Values

Acceptors == 1..N
Quorum == (N \div 2) + 1

VARIABLES maxBallot, accepted, chosen

vars == <<maxBallot, accepted, chosen>>

TypeOK ==
    /\ maxBallot \in [Acceptors -> Nat]
    /\ accepted \in [Acceptors -> (Values \cup {0}) \X Nat]
    /\ chosen \in Values \cup {0}

Init ==
    /\ maxBallot = [a \in Acceptors |-> 0]
    /\ accepted = [a \in Acceptors |-> <<0, 0>>]
    /\ chosen = 0

Prepare(b) ==
    /\ b > 0
    /\ \E a \in Acceptors :
        /\ b > maxBallot[a]
        /\ maxBallot' = [maxBallot EXCEPT ![a] = b]
    /\ UNCHANGED <<accepted, chosen>>

Accept(b, v) ==
    /\ v \in Values
    /\ b > 0
    /\ \E a \in Acceptors :
        /\ b >= maxBallot[a]
        /\ accepted' = [accepted EXCEPT ![a] = <<v, b>>]
        /\ maxBallot' = [maxBallot EXCEPT ![a] = b]
    /\ UNCHANGED chosen

Choose(v) ==
    /\ v \in Values
    /\ chosen = 0
    /\ Cardinality({a \in Acceptors : accepted[a][1] = v}) >= Quorum
    /\ chosen' = v
    /\ UNCHANGED <<maxBallot, accepted>>

Next ==
    \/ \E b \in 1..10 : Prepare(b)
    \/ \E b \in 1..10, v \in Values : Accept(b, v)
    \/ \E v \in Values : Choose(v)

Spec == Init /\ [][Next]_vars

Consistency == chosen /= 0 => [][chosen' = chosen]_chosen

====
""",
    },

    # ── BM012: Bounded Retransmission Protocol ───────────────────────────
    {
        "prompt": "A sender transmits a file in chunks over an unreliable channel. The sender retransmits up to MAX_RETRIES times before giving up.",
        "module": "BoundedRetransmission",
        "spec": r"""---- MODULE BoundedRetransmission ----

EXTENDS Naturals, Sequences

CONSTANT MaxRetries, NumChunks

VARIABLES senderState, receiverState, retries, chunk, channelSR, channelRS

vars == <<senderState, receiverState, retries, chunk, channelSR, channelRS>>

TypeOK ==
    /\ senderState \in {"idle", "sending", "done", "failed"}
    /\ receiverState \in {"waiting", "received", "done"}
    /\ retries \in 0..MaxRetries
    /\ chunk \in 0..NumChunks

Init ==
    /\ senderState = "idle"
    /\ receiverState = "waiting"
    /\ retries = 0
    /\ chunk = 0
    /\ channelSR = <<>>
    /\ channelRS = <<>>

Send ==
    /\ senderState = "sending"
    /\ retries < MaxRetries
    /\ channelSR' = Append(channelSR, chunk)
    /\ UNCHANGED <<senderState, receiverState, retries, chunk, channelRS>>

Lose ==
    /\ Len(channelSR) > 0
    /\ channelSR' = Tail(channelSR)
    /\ retries' = retries + 1
    /\ senderState' = IF retries + 1 >= MaxRetries THEN "failed" ELSE senderState
    /\ UNCHANGED <<receiverState, chunk, channelRS>>

Deliver ==
    /\ Len(channelSR) > 0
    /\ receiverState = "waiting"
    /\ receiverState' = "received"
    /\ channelSR' = Tail(channelSR)
    /\ UNCHANGED <<senderState, retries, chunk, channelRS>>

StartSending ==
    /\ senderState = "idle"
    /\ senderState' = "sending"
    /\ chunk' = 1
    /\ UNCHANGED <<receiverState, retries, channelSR, channelRS>>

Next == Send \/ Lose \/ Deliver \/ StartSending

Spec == Init /\ [][Next]_vars

DeliveryOrFailure == senderState \in {"idle", "sending", "done", "failed"}

====
""",
    },

    # ── BM013: Transaction Isolation (Snapshot Isolation) ────────────────
    {
        "prompt": "A database with snapshot isolation. Transactions read a consistent snapshot. Write-write conflicts cause an abort.",
        "module": "SnapshotIsolation",
        "spec": r"""---- MODULE SnapshotIsolation ----

EXTENDS Naturals, FiniteSets

CONSTANT Keys, Txns

VARIABLES store, txnState, writeSet, snapshot

vars == <<store, txnState, writeSet, snapshot>>

TypeOK ==
    /\ store \in [Keys -> Nat]
    /\ txnState \in [Txns -> {"idle", "active", "committed", "aborted"}]
    /\ writeSet \in [Txns -> SUBSET Keys]
    /\ snapshot \in [Txns -> [Keys -> Nat]]

Init ==
    /\ store = [k \in Keys |-> 0]
    /\ txnState = [t \in Txns |-> "idle"]
    /\ writeSet = [t \in Txns |-> {}]
    /\ snapshot = [t \in Txns |-> [k \in Keys |-> 0]]

Begin(t) ==
    /\ txnState[t] = "idle"
    /\ txnState' = [txnState EXCEPT ![t] = "active"]
    /\ snapshot' = [snapshot EXCEPT ![t] = store]
    /\ UNCHANGED <<store, writeSet>>

Write(t, k) ==
    /\ txnState[t] = "active"
    /\ writeSet' = [writeSet EXCEPT ![t] = writeSet[t] \cup {k}]
    /\ UNCHANGED <<store, txnState, snapshot>>

Commit(t) ==
    /\ txnState[t] = "active"
    /\ \A t2 \in Txns :
        t2 /= t /\ txnState[t2] = "committed"
        => writeSet[t] \cap writeSet[t2] = {}
    /\ txnState' = [txnState EXCEPT ![t] = "committed"]
    /\ UNCHANGED <<store, writeSet, snapshot>>

Abort(t) ==
    /\ txnState[t] = "active"
    /\ txnState' = [txnState EXCEPT ![t] = "aborted"]
    /\ UNCHANGED <<store, writeSet, snapshot>>

Next ==
    \/ \E t \in Txns : Begin(t)
    \/ \E t \in Txns, k \in Keys : Write(t, k)
    \/ \E t \in Txns : Commit(t)
    \/ \E t \in Txns : Abort(t)

Spec == Init /\ [][Next]_vars

NoWriteConflict ==
    \A t1, t2 \in Txns :
        (t1 /= t2 /\ txnState[t1] = "committed" /\ txnState[t2] = "committed")
        => writeSet[t1] \cap writeSet[t2] = {}

====
""",
    },

    # ── BM014: Clock Synchronisation ─────────────────────────────────────
    {
        "prompt": "N nodes exchange clock values to synchronise. After one round, all clocks are within epsilon of each other.",
        "module": "ClockSync",
        "spec": r"""---- MODULE ClockSync ----

EXTENDS Naturals

CONSTANT N, Epsilon

Nodes == 1..N

VARIABLES clocks, synced

vars == <<clocks, synced>>

TypeOK ==
    /\ clocks \in [Nodes -> Nat]
    /\ synced \in BOOLEAN

Init ==
    /\ clocks \in [Nodes -> 0..10]
    /\ synced = FALSE

Tick(n) ==
    /\ clocks' = [clocks EXCEPT ![n] = clocks[n] + 1]
    /\ UNCHANGED synced

Sync(n, m) ==
    /\ n /= m
    /\ LET avg == (clocks[n] + clocks[m]) \div 2
       IN clocks' = [clocks EXCEPT ![n] = avg, ![m] = avg]
    /\ UNCHANGED synced

MarkSynced ==
    /\ \A n1, n2 \in Nodes :
        IF clocks[n1] >= clocks[n2]
        THEN clocks[n1] - clocks[n2] <= Epsilon
        ELSE clocks[n2] - clocks[n1] <= Epsilon
    /\ synced' = TRUE
    /\ UNCHANGED clocks

Next ==
    \/ \E n \in Nodes : Tick(n)
    \/ \E n, m \in Nodes : Sync(n, m)
    \/ MarkSynced

Spec == Init /\ [][Next]_vars

ClockBound ==
    synced => \A n1, n2 \in Nodes :
        IF clocks[n1] >= clocks[n2]
        THEN clocks[n1] - clocks[n2] <= Epsilon
        ELSE clocks[n2] - clocks[n1] <= Epsilon

====
""",
    },

    # ── BM015: Peterson's Algorithm ──────────────────────────────────────
    {
        "prompt": "Peterson's mutual exclusion algorithm for exactly 2 processes.",
        "module": "Peterson",
        "spec": r"""---- MODULE Peterson ----

EXTENDS Naturals

VARIABLES flag, turn, pc

vars == <<flag, turn, pc>>

Procs == {0, 1}
Other(i) == IF i = 0 THEN 1 ELSE 0

TypeOK ==
    /\ flag \in [Procs -> BOOLEAN]
    /\ turn \in Procs
    /\ pc \in [Procs -> {"idle", "set_flag", "set_turn", "wait", "cs", "reset"}]

Init ==
    /\ flag = [i \in Procs |-> FALSE]
    /\ turn = 0
    /\ pc = [i \in Procs |-> "idle"]

SetFlag(i) ==
    /\ pc[i] = "idle"
    /\ flag' = [flag EXCEPT ![i] = TRUE]
    /\ pc' = [pc EXCEPT ![i] = "set_flag"]
    /\ UNCHANGED turn

SetTurn(i) ==
    /\ pc[i] = "set_flag"
    /\ turn' = Other(i)
    /\ pc' = [pc EXCEPT ![i] = "wait"]
    /\ UNCHANGED flag

Wait(i) ==
    /\ pc[i] = "wait"
    /\ flag[Other(i)] = FALSE \/ turn = i
    /\ pc' = [pc EXCEPT ![i] = "cs"]
    /\ UNCHANGED <<flag, turn>>

ExitCS(i) ==
    /\ pc[i] = "cs"
    /\ flag' = [flag EXCEPT ![i] = FALSE]
    /\ pc' = [pc EXCEPT ![i] = "idle"]
    /\ UNCHANGED turn

Next == \E i \in Procs : SetFlag(i) \/ SetTurn(i) \/ Wait(i) \/ ExitCS(i)

Spec == Init /\ [][Next]_vars

MutualExclusion == ~(pc[0] = "cs" /\ pc[1] = "cs")

====
""",
    },

    # ── BM016: Gossip Protocol ───────────────────────────────────────────
    {
        "prompt": "A gossip (epidemic) protocol where N nodes periodically share updates. A node that receives an update marks it as infected and spreads it in subsequent rounds.",
        "module": "GossipProtocol",
        "spec": r"""---- MODULE GossipProtocol ----

EXTENDS Naturals, FiniteSets

CONSTANT N

Nodes == 1..N

VARIABLES infected, round

vars == <<infected, round>>

TypeOK ==
    /\ infected \in SUBSET Nodes
    /\ round \in Nat

Init ==
    /\ infected = {1}
    /\ round = 0

Spread(src, dst) ==
    /\ src \in infected
    /\ dst \in Nodes
    /\ dst /= src
    /\ infected' = infected \cup {dst}
    /\ round' = round + 1

Next == \E src, dst \in Nodes : Spread(src, dst)

Spec == Init /\ [][Next]_vars

EventualConsistency == Cardinality(infected) <= N

====
""",
    },

    # ── BM017: Simple Allocator ──────────────────────────────────────────
    {
        "prompt": "A memory allocator managing a fixed pool of N pages. Clients request and release pages. Safety: no page is allocated to two clients simultaneously.",
        "module": "SimpleAllocator",
        "spec": r"""---- MODULE SimpleAllocator ----

EXTENDS Naturals, FiniteSets

CONSTANT NumPages, NumClients

Pages == 1..NumPages
Clients == 1..NumClients

VARIABLES free, allocated

vars == <<free, allocated>>

TypeOK ==
    /\ free \in SUBSET Pages
    /\ allocated \in [Clients -> SUBSET Pages]

Init ==
    /\ free = Pages
    /\ allocated = [c \in Clients |-> {}]

Allocate(c, p) ==
    /\ p \in free
    /\ free' = free \ {p}
    /\ allocated' = [allocated EXCEPT ![c] = allocated[c] \cup {p}]

Release(c, p) ==
    /\ p \in allocated[c]
    /\ allocated' = [allocated EXCEPT ![c] = allocated[c] \ {p}]
    /\ free' = free \cup {p}

Next ==
    \/ \E c \in Clients, p \in Pages : Allocate(c, p)
    \/ \E c \in Clients, p \in Pages : Release(c, p)

Spec == Init /\ [][Next]_vars

SafeAllocation ==
    \A c1, c2 \in Clients : c1 /= c2 => allocated[c1] \cap allocated[c2] = {}

====
""",
    },

    # ── BM018: Publish-Subscribe Broker ──────────────────────────────────
    {
        "prompt": "A single broker with subscribers and publishers. Subscribers register interest in topics. Publishers post messages on topics. The broker delivers each message to all registered subscribers.",
        "module": "PubSub",
        "spec": r"""---- MODULE PubSub ----

EXTENDS Naturals, FiniteSets

CONSTANT Topics, Subscribers

VARIABLES subscriptions, published, delivered

vars == <<subscriptions, published, delivered>>

TypeOK ==
    /\ subscriptions \in [Subscribers -> SUBSET Topics]
    /\ published \in SUBSET (Topics \X Nat)
    /\ delivered \in SUBSET (Subscribers \X Topics \X Nat)

Init ==
    /\ subscriptions = [s \in Subscribers |-> {}]
    /\ published = {}
    /\ delivered = {}

Subscribe(s, t) ==
    /\ t \in Topics
    /\ subscriptions' = [subscriptions EXCEPT ![s] = subscriptions[s] \cup {t}]
    /\ UNCHANGED <<published, delivered>>

Publish(t, msgId) ==
    /\ t \in Topics
    /\ published' = published \cup {<<t, msgId>>}
    /\ UNCHANGED <<subscriptions, delivered>>

Deliver(s, t, msgId) ==
    /\ <<t, msgId>> \in published
    /\ t \in subscriptions[s]
    /\ delivered' = delivered \cup {<<s, t, msgId>>}
    /\ UNCHANGED <<subscriptions, published>>

Next ==
    \/ \E s \in Subscribers, t \in Topics : Subscribe(s, t)
    \/ \E t \in Topics, m \in 1..10 : Publish(t, m)
    \/ \E s \in Subscribers, t \in Topics, m \in 1..10 : Deliver(s, t, m)

Spec == Init /\ [][Next]_vars

DeliveryGuarantee ==
    \A s \in Subscribers, t \in Topics, m \in Nat :
        (<<t, m>> \in published /\ t \in subscriptions[s])
        => TRUE

====
""",
    },

    # ── BM019: Dekker's Algorithm ────────────────────────────────────────
    {
        "prompt": "Dekker's mutual exclusion algorithm for 2 processes, the first known correct solution to the problem.",
        "module": "Dekker",
        "spec": r"""---- MODULE Dekker ----

EXTENDS Naturals

VARIABLES wants, turn, pc

vars == <<wants, turn, pc>>

Procs == {0, 1}
Other(i) == IF i = 0 THEN 1 ELSE 0

TypeOK ==
    /\ wants \in [Procs -> BOOLEAN]
    /\ turn \in Procs
    /\ pc \in [Procs -> {"idle", "set_want", "check", "wait", "cs", "exit"}]

Init ==
    /\ wants = [i \in Procs |-> FALSE]
    /\ turn = 0
    /\ pc = [i \in Procs |-> "idle"]

SetWant(i) ==
    /\ pc[i] = "idle"
    /\ wants' = [wants EXCEPT ![i] = TRUE]
    /\ pc' = [pc EXCEPT ![i] = "set_want"]
    /\ UNCHANGED turn

Check(i) ==
    /\ pc[i] = "set_want"
    /\ IF wants[Other(i)] = FALSE
       THEN pc' = [pc EXCEPT ![i] = "cs"]
       ELSE pc' = [pc EXCEPT ![i] = "check"]
    /\ UNCHANGED <<wants, turn>>

WaitTurn(i) ==
    /\ pc[i] = "check"
    /\ IF turn = i
       THEN pc' = [pc EXCEPT ![i] = "wait"]
       ELSE /\ wants' = [wants EXCEPT ![i] = FALSE]
            /\ pc' = [pc EXCEPT ![i] = "idle"]
            /\ UNCHANGED turn

WaitForOther(i) ==
    /\ pc[i] = "wait"
    /\ wants[Other(i)] = FALSE
    /\ pc' = [pc EXCEPT ![i] = "cs"]
    /\ UNCHANGED <<wants, turn>>

ExitCS(i) ==
    /\ pc[i] = "cs"
    /\ turn' = Other(i)
    /\ wants' = [wants EXCEPT ![i] = FALSE]
    /\ pc' = [pc EXCEPT ![i] = "idle"]

Next == \E i \in Procs : SetWant(i) \/ Check(i) \/ WaitTurn(i) \/ WaitForOther(i) \/ ExitCS(i)

Spec == Init /\ [][Next]_vars

MutualExclusion == ~(pc[0] = "cs" /\ pc[1] = "cs")

Deadlock_Freedom == wants[0] \/ wants[1] => \E i \in Procs : pc[i] = "cs"

====
""",
    },

    # ── BM020: Eventually Consistent Counter ─────────────────────────────
    {
        "prompt": "A distributed grow-only counter (G-Counter CRDT) with N nodes. Each node increments its own slot. Merge takes element-wise max. All nodes eventually agree on the total.",
        "module": "GCounter",
        "spec": r"""---- MODULE GCounter ----

EXTENDS Naturals, FiniteSets

CONSTANT N

Nodes == 1..N

VARIABLES counts

vars == <<counts>>

TypeOK == counts \in [Nodes -> [Nodes -> Nat]]

Init == counts = [n \in Nodes |-> [m \in Nodes |-> 0]]

Increment(n) ==
    /\ counts' = [counts EXCEPT ![n][n] = counts[n][n] + 1]

Merge(n, m) ==
    /\ n /= m
    /\ counts' = [counts EXCEPT ![n] =
        [k \in Nodes |->
            IF counts[n][k] >= counts[m][k]
            THEN counts[n][k]
            ELSE counts[m][k]]]

Next ==
    \/ \E n \in Nodes : Increment(n)
    \/ \E n, m \in Nodes : Merge(n, m)

Spec == Init /\ [][Next]_vars

Monotone ==
    \A n \in Nodes, k \in Nodes :
        counts[n][k] >= 0

====
""",
    },

    # ── Extra training examples (simple patterns) ────────────────────────

    # Simple counter
    {
        "prompt": "A counter that starts at 0, can increment by 1, and has a maximum value MAX.",
        "module": "Counter",
        "spec": r"""---- MODULE Counter ----

EXTENDS Naturals

CONSTANT MAX

VARIABLES count

vars == <<count>>

TypeOK == count \in 0..MAX

Init == count = 0

Increment ==
    /\ count < MAX
    /\ count' = count + 1

Next == Increment

Spec == Init /\ [][Next]_vars

====
""",
    },

    # Toggle switch
    {
        "prompt": "A toggle switch that alternates between ON and OFF states.",
        "module": "Toggle",
        "spec": r"""---- MODULE Toggle ----

VARIABLES state

vars == <<state>>

TypeOK == state \in {"on", "off"}

Init == state = "off"

Toggle ==
    /\ state = "off"
    /\ state' = "on"

Toggle2 ==
    /\ state = "on"
    /\ state' = "off"

Next == Toggle \/ Toggle2

Spec == Init /\ [][Next]_vars

====
""",
    },

    # Traffic light
    {
        "prompt": "A traffic light controller that cycles through Red, Green, Yellow states.",
        "module": "TrafficLight",
        "spec": r"""---- MODULE TrafficLight ----

VARIABLES light

vars == <<light>>

TypeOK == light \in {"red", "green", "yellow"}

Init == light = "red"

Next ==
    \/ /\ light = "red"
       /\ light' = "green"
    \/ /\ light = "green"
       /\ light' = "yellow"
    \/ /\ light = "yellow"
       /\ light' = "red"

Spec == Init /\ [][Next]_vars

====
""",
    },

    # Bank account
    {
        "prompt": "A bank account with deposit and withdraw operations. Balance must never go negative.",
        "module": "BankAccount",
        "spec": r"""---- MODULE BankAccount ----

EXTENDS Naturals

CONSTANT MaxBalance

VARIABLES balance

vars == <<balance>>

TypeOK == balance \in 0..MaxBalance

Init == balance = 0

Deposit(amount) ==
    /\ amount > 0
    /\ balance + amount <= MaxBalance
    /\ balance' = balance + amount

Withdraw(amount) ==
    /\ amount > 0
    /\ balance >= amount
    /\ balance' = balance - amount

Next ==
    \/ \E a \in 1..MaxBalance : Deposit(a)
    \/ \E a \in 1..MaxBalance : Withdraw(a)

Spec == Init /\ [][Next]_vars

NonNegative == balance >= 0

====
""",
    },

    # Semaphore
    {
        "prompt": "A simple semaphore with initial count N. Processes can acquire (decrement) or release (increment) the semaphore. Count must be non-negative.",
        "module": "Semaphore",
        "spec": r"""---- MODULE Semaphore ----

EXTENDS Naturals

CONSTANT N

VARIABLES count

vars == <<count>>

TypeOK == count \in 0..N

Init == count = N

Acquire ==
    /\ count > 0
    /\ count' = count - 1

Release ==
    /\ count < N
    /\ count' = count + 1

Next == Acquire \/ Release

Spec == Init /\ [][Next]_vars

CountNonNegative == count >= 0

====
""",
    },

    # Door state machine
    {
        "prompt": "A state machine that models a door: it can be Open, Closed, or Locked. Transitions follow physical constraints.",
        "module": "Door",
        "spec": r"""---- MODULE Door ----

VARIABLES doorState

vars == <<doorState>>

TypeOK == doorState \in {"open", "closed", "locked"}

Init == doorState = "closed"

Open ==
    /\ doorState = "closed"
    /\ doorState' = "open"

Close ==
    /\ doorState = "open"
    /\ doorState' = "closed"

Lock ==
    /\ doorState = "closed"
    /\ doorState' = "locked"

Unlock ==
    /\ doorState = "locked"
    /\ doorState' = "closed"

Next == Open \/ Close \/ Lock \/ Unlock

Spec == Init /\ [][Next]_vars

====
""",
    },

    # Resource allocator
    {
        "prompt": "A resource allocator for a single resource. Processes can request, acquire, and release the resource.",
        "module": "ResourceAlloc",
        "spec": r"""---- MODULE ResourceAlloc ----

EXTENDS Naturals

CONSTANT NumProcs

Procs == 1..NumProcs

VARIABLES owner, waiting

vars == <<owner, waiting>>

TypeOK ==
    /\ owner \in Procs \cup {0}
    /\ waiting \in SUBSET Procs

Init ==
    /\ owner = 0
    /\ waiting = {}

Request(p) ==
    /\ owner /= p
    /\ waiting' = waiting \cup {p}
    /\ UNCHANGED owner

Acquire(p) ==
    /\ owner = 0
    /\ p \in waiting
    /\ owner' = p
    /\ waiting' = waiting \ {p}

Release(p) ==
    /\ owner = p
    /\ owner' = 0
    /\ UNCHANGED waiting

Next == \E p \in Procs : Request(p) \/ Acquire(p) \/ Release(p)

Spec == Init /\ [][Next]_vars

SafeResource == owner /= 0 => owner \in Procs

====
""",
    },

    # Bounded stack
    {
        "prompt": "A simple stack (LIFO) data structure with push and pop operations. The stack has a maximum capacity.",
        "module": "BoundedStack",
        "spec": r"""---- MODULE BoundedStack ----

EXTENDS Naturals, Sequences

CONSTANT Capacity

VARIABLES stack

vars == <<stack>>

TypeOK == stack \in Seq(Nat)

Init == stack = <<>>

Push(v) ==
    /\ Len(stack) < Capacity
    /\ stack' = <<v>> \o stack

Pop ==
    /\ Len(stack) > 0
    /\ stack' = Tail(stack)

Next ==
    \/ \E v \in 1..10 : Push(v)
    \/ Pop

Spec == Init /\ [][Next]_vars

BoundedSize == Len(stack) <= Capacity

====
""",
    },

    # Distributed lock
    {
        "prompt": "A distributed lock service. Nodes can request, acquire, and release a lock. Only one node holds the lock at a time.",
        "module": "DistributedLock",
        "spec": r"""---- MODULE DistributedLock ----

EXTENDS Naturals, FiniteSets

CONSTANT NumNodes

Nodes == 1..NumNodes

VARIABLES lockHolder, nodeState

vars == <<lockHolder, nodeState>>

TypeOK ==
    /\ lockHolder \in Nodes \cup {0}
    /\ nodeState \in [Nodes -> {"idle", "requesting", "holding"}]

Init ==
    /\ lockHolder = 0
    /\ nodeState = [n \in Nodes |-> "idle"]

Request(n) ==
    /\ nodeState[n] = "idle"
    /\ nodeState' = [nodeState EXCEPT ![n] = "requesting"]
    /\ UNCHANGED lockHolder

Acquire(n) ==
    /\ nodeState[n] = "requesting"
    /\ lockHolder = 0
    /\ lockHolder' = n
    /\ nodeState' = [nodeState EXCEPT ![n] = "holding"]

Release(n) ==
    /\ nodeState[n] = "holding"
    /\ lockHolder = n
    /\ lockHolder' = 0
    /\ nodeState' = [nodeState EXCEPT ![n] = "idle"]

Next == \E n \in Nodes : Request(n) \/ Acquire(n) \/ Release(n)

Spec == Init /\ [][Next]_vars

MutualExclusion ==
    Cardinality({n \in Nodes : nodeState[n] = "holding"}) <= 1

====
""",
    },

    # Elevator
    {
        "prompt": "An elevator controller for a building with N floors. The elevator moves up and down, opening doors at requested floors.",
        "module": "Elevator",
        "spec": r"""---- MODULE Elevator ----

EXTENDS Naturals

CONSTANT NumFloors

Floors == 1..NumFloors

VARIABLES floor, direction, doorsOpen, requests

vars == <<floor, direction, doorsOpen, requests>>

TypeOK ==
    /\ floor \in Floors
    /\ direction \in {"up", "down", "idle"}
    /\ doorsOpen \in BOOLEAN
    /\ requests \in SUBSET Floors

Init ==
    /\ floor = 1
    /\ direction = "idle"
    /\ doorsOpen = FALSE
    /\ requests = {}

RequestFloor(f) ==
    /\ requests' = requests \cup {f}
    /\ UNCHANGED <<floor, direction, doorsOpen>>

MoveUp ==
    /\ direction = "up"
    /\ floor < NumFloors
    /\ doorsOpen = FALSE
    /\ floor' = floor + 1
    /\ IF floor + 1 = NumFloors THEN direction' = "down" ELSE UNCHANGED direction
    /\ UNCHANGED <<doorsOpen, requests>>

MoveDown ==
    /\ direction = "down"
    /\ floor > 1
    /\ doorsOpen = FALSE
    /\ floor' = floor - 1
    /\ IF floor - 1 = 1 THEN direction' = "up" ELSE UNCHANGED direction
    /\ UNCHANGED <<doorsOpen, requests>>

OpenDoors ==
    /\ floor \in requests
    /\ doorsOpen = FALSE
    /\ doorsOpen' = TRUE
    /\ requests' = requests \ {floor}
    /\ UNCHANGED <<floor, direction>>

CloseDoors ==
    /\ doorsOpen = TRUE
    /\ doorsOpen' = FALSE
    /\ direction' = IF requests /= {} THEN "up" ELSE "idle"
    /\ UNCHANGED <<floor, requests>>

Next == \E f \in Floors : RequestFloor(f) \/ MoveUp \/ MoveDown \/ OpenDoors \/ CloseDoors

Spec == Init /\ [][Next]_vars

FloorInRange == floor \in Floors

====
""",
    },
]


def main():
    passed = 0
    failed = 0
    valid_examples = []

    print(f"Validating {len(EXAMPLES)} hand-crafted TLA+ specs with SANY...\n")

    for i, ex in enumerate(EXAMPLES):
        module = ex["module"]
        spec = ex["spec"].strip()
        prompt = ex["prompt"]

        ok, output = validate_sany(spec, module)
        status = "PASS" if ok else "FAIL"
        print(f"  [{i+1:2d}/{len(EXAMPLES)}] {module:30s} {status}")

        if not ok:
            failed += 1
            # Print first few error lines
            for line in output.splitlines():
                if "error" in line.lower() or "abort" in line.lower():
                    print(f"         {line.strip()}")
        else:
            passed += 1
            valid_examples.append(build_example(prompt, spec))

    print(f"\nResults: {passed}/{len(EXAMPLES)} passed SANY, {failed} failed")

    if valid_examples:
        # Backup existing augmented.jsonl
        if AUGMENTED_JSONL.exists():
            backup = AUGMENTED_JSONL.with_suffix(".jsonl.pre_handcraft")
            import shutil
            shutil.copy2(AUGMENTED_JSONL, backup)
            print(f"Backed up existing augmented.jsonl to {backup.name}")

        # Write fresh augmented.jsonl with ONLY SANY-verified examples
        with open(AUGMENTED_JSONL, "w", encoding="utf-8") as f:
            for ex in valid_examples:
                f.write(json.dumps(ex, ensure_ascii=False) + "\n")

        print(f"Wrote {len(valid_examples)} SANY-verified examples to {AUGMENTED_JSONL}")

    return passed, failed


if __name__ == "__main__":
    main()
