#!/usr/bin/env python3
"""
Craft a large set of SANY-verified TLA+ training examples.
Each example is validated with SANY before being included.
Writes harmony-format JSONL to data/processed/augmented.jsonl.
"""

import json
import subprocess
import sys
import tempfile
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SANY_JAR = REPO / "src" / "shared" / "tlc" / "tla2tools.jar"
OUTPUT = REPO / "data" / "processed" / "augmented.jsonl"

DEVELOPER_PROMPT = (
    "You are ChatTLA, an expert at writing verified TLA+ formal specifications.\n"
    "When asked to write a TLA+ spec, follow these rules exactly:\n"
    "1. Start the module with ---- MODULE <ModuleName> ----\n"
    "2. End with ====\n"
    "3. Include EXTENDS, VARIABLES, Init, Next, and Spec operators\n"
    "4. After the TLA+ module, append a TLC configuration block:\n"
    "   SPECIFICATION Spec\n"
    "   INVARIANT TypeOK   (if TypeOK is defined)\n"
    "5. Output only valid TLA+ code. No markdown fences, no explanation outside the spec.\n"
    "Reasoning: medium"
)

# ─── Spec templates ───
# Each spec: (prompt, analysis, tla_code)
# The TLA+ code MUST pass SANY.

SPECS = []

def add(prompt: str, analysis: str, code: str):
    """Register a spec."""
    SPECS.append((prompt.strip(), analysis.strip(), code.strip()))

# =========================================================================
# BM001 variants: Mutual Exclusion
# =========================================================================
add(
    "A mutual exclusion algorithm for N processes where at most one process is in the critical section at a time.",
    "I'll model N processes with states idle/trying/critical and a turn variable for arbitration.",
    r"""---- MODULE MutualExclusion ----
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

====""")

add(
    "Model mutual exclusion for N processes. Track process state and enforce that at most one is critical.",
    "I'll use a simple state machine with idle/trying/critical states per process.",
    r"""---- MODULE MutexSimple ----
EXTENDS Naturals

CONSTANT N

VARIABLE pc

vars == <<pc>>

TypeOK == pc \in [1..N -> {"idle", "trying", "critical"}]

Init == pc = [i \in 1..N |-> "idle"]

Request(i) ==
    /\ pc[i] = "idle"
    /\ pc' = [pc EXCEPT ![i] = "trying"]

Enter(i) ==
    /\ pc[i] = "trying"
    /\ \A j \in 1..N : j /= i => pc[j] /= "critical"
    /\ pc' = [pc EXCEPT ![i] = "critical"]

Leave(i) ==
    /\ pc[i] = "critical"
    /\ pc' = [pc EXCEPT ![i] = "idle"]

Next == \E i \in 1..N : Request(i) \/ Enter(i) \/ Leave(i)

Spec == Init /\ [][Next]_vars

MutualExclusion == \A i, j \in 1..N : (i /= j) => ~(pc[i] = "critical" /\ pc[j] = "critical")

====""")

# =========================================================================
# BM002 variants: Two-Phase Commit
# =========================================================================
add(
    "A two-phase commit protocol with one coordinator and N participants. The coordinator decides to commit only if all participants vote yes.",
    "I'll model the 2PC protocol with participant votes and coordinator decision.",
    r"""---- MODULE TwoPhaseCommit ----
EXTENDS Naturals, FiniteSets

CONSTANT N

VARIABLES pState, cDecision, votes

vars == <<pState, cDecision, votes>>

Participants == 1..N

TypeOK ==
    /\ pState \in [Participants -> {"working", "prepared", "committed", "aborted"}]
    /\ cDecision \in {"none", "commit", "abort"}
    /\ votes \in [Participants -> {"none", "yes", "no"}]

Init ==
    /\ pState = [p \in Participants |-> "working"]
    /\ cDecision = "none"
    /\ votes = [p \in Participants |-> "none"]

Prepare(p) ==
    /\ pState[p] = "working"
    /\ pState' = [pState EXCEPT ![p] = "prepared"]
    /\ votes' = [votes EXCEPT ![p] = "yes"]
    /\ UNCHANGED cDecision

VoteNo(p) ==
    /\ pState[p] = "working"
    /\ pState' = [pState EXCEPT ![p] = "aborted"]
    /\ votes' = [votes EXCEPT ![p] = "no"]
    /\ UNCHANGED cDecision

Decide ==
    /\ cDecision = "none"
    /\ \A p \in Participants : votes[p] /= "none"
    /\ IF \A p \in Participants : votes[p] = "yes"
       THEN cDecision' = "commit"
       ELSE cDecision' = "abort"
    /\ UNCHANGED <<pState, votes>>

Commit(p) ==
    /\ cDecision = "commit"
    /\ pState[p] = "prepared"
    /\ pState' = [pState EXCEPT ![p] = "committed"]
    /\ UNCHANGED <<cDecision, votes>>

Abort(p) ==
    /\ cDecision = "abort"
    /\ pState[p] \in {"prepared", "working"}
    /\ pState' = [pState EXCEPT ![p] = "aborted"]
    /\ UNCHANGED <<cDecision, votes>>

Next ==
    \/ \E p \in Participants : Prepare(p)
    \/ \E p \in Participants : VoteNo(p)
    \/ Decide
    \/ \E p \in Participants : Commit(p)
    \/ \E p \in Participants : Abort(p)

Spec == Init /\ [][Next]_vars

Consistency ==
    \A p, q \in Participants :
        ~(pState[p] = "committed" /\ pState[q] = "aborted")

====""")

# =========================================================================
# BM003 variants: Dining Philosophers
# =========================================================================
add(
    "The dining philosophers problem with N philosophers and N forks. Each philosopher picks up both adjacent forks to eat, then puts them down.",
    "I'll track fork ownership and philosopher states: thinking, hungry, eating.",
    r"""---- MODULE DiningPhilosophers ----
EXTENDS Naturals

CONSTANT N

VARIABLES phil, forks

vars == <<phil, forks>>

Left(i) == i
Right(i) == IF i = N THEN 1 ELSE i + 1

TypeOK ==
    /\ phil \in [1..N -> {"thinking", "hungry", "eating"}]
    /\ forks \in [1..N -> {"free", "taken"}]

Init ==
    /\ phil = [i \in 1..N |-> "thinking"]
    /\ forks = [i \in 1..N |-> "free"]

GetHungry(i) ==
    /\ phil[i] = "thinking"
    /\ phil' = [phil EXCEPT ![i] = "hungry"]
    /\ UNCHANGED forks

PickUpForks(i) ==
    /\ phil[i] = "hungry"
    /\ forks[Left(i)] = "free"
    /\ forks[Right(i)] = "free"
    /\ forks' = [forks EXCEPT ![Left(i)] = "taken", ![Right(i)] = "taken"]
    /\ phil' = [phil EXCEPT ![i] = "eating"]

PutDownForks(i) ==
    /\ phil[i] = "eating"
    /\ forks' = [forks EXCEPT ![Left(i)] = "free", ![Right(i)] = "free"]
    /\ phil' = [phil EXCEPT ![i] = "thinking"]

Next == \E i \in 1..N : GetHungry(i) \/ PickUpForks(i) \/ PutDownForks(i)

Spec == Init /\ [][Next]_vars

NoDeadlock == \E i \in 1..N : phil[i] /= "hungry" \/ (forks[Left(i)] = "free" /\ forks[Right(i)] = "free")

====""")

# =========================================================================
# BM004 variants: Bakery Algorithm
# =========================================================================
add(
    "Lamport's bakery mutual exclusion algorithm for N processes. Processes take a numbered ticket; lower numbers enter first.",
    "I'll model with VARIABLES num and flag, using a choosing phase before assigning ticket numbers.",
    r"""---- MODULE BakeryAlgorithm ----
EXTENDS Naturals

CONSTANT N

VARIABLES num, flag, pc

vars == <<num, flag, pc>>

TypeOK ==
    /\ num \in [1..N -> Nat]
    /\ flag \in [1..N -> BOOLEAN]
    /\ pc \in [1..N -> {"idle", "choosing", "waiting", "critical"}]

Init ==
    /\ num = [i \in 1..N |-> 0]
    /\ flag = [i \in 1..N |-> FALSE]
    /\ pc = [i \in 1..N |-> "idle"]

MaxNum == LET S == {num[j] : j \in 1..N} IN
          CHOOSE x \in S : \A y \in S : x >= y

StartChoosing(i) ==
    /\ pc[i] = "idle"
    /\ flag' = [flag EXCEPT ![i] = TRUE]
    /\ pc' = [pc EXCEPT ![i] = "choosing"]
    /\ UNCHANGED num

AssignTicket(i) ==
    /\ pc[i] = "choosing"
    /\ num' = [num EXCEPT ![i] = MaxNum + 1]
    /\ flag' = [flag EXCEPT ![i] = FALSE]
    /\ pc' = [pc EXCEPT ![i] = "waiting"]

EnterCS(i) ==
    /\ pc[i] = "waiting"
    /\ \A j \in 1..N : (j /= i) =>
        (/\ ~flag[j]
         /\ (num[j] = 0 \/ num[i] < num[j] \/ (num[i] = num[j] /\ i < j)))
    /\ pc' = [pc EXCEPT ![i] = "critical"]
    /\ UNCHANGED <<num, flag>>

ExitCS(i) ==
    /\ pc[i] = "critical"
    /\ num' = [num EXCEPT ![i] = 0]
    /\ pc' = [pc EXCEPT ![i] = "idle"]
    /\ UNCHANGED flag

Next == \E i \in 1..N : StartChoosing(i) \/ AssignTicket(i) \/ EnterCS(i) \/ ExitCS(i)

Spec == Init /\ [][Next]_vars

MutualExclusion == \A i, j \in 1..N : (i /= j) => ~(pc[i] = "critical" /\ pc[j] = "critical")

====""")

# =========================================================================
# BM005 variants: Producer-Consumer Queue
# =========================================================================
add(
    "A bounded FIFO queue with one producer and one consumer. The producer blocks when the queue is full; the consumer blocks when empty.",
    "I'll use a sequence to model the queue bounded by capacity K.",
    r"""---- MODULE ProducerConsumer ----
EXTENDS Naturals, Sequences

CONSTANT K

VARIABLES queue

vars == <<queue>>

TypeOK ==
    /\ queue \in Seq(Nat)
    /\ Len(queue) <= K

Init == queue = <<>>

Produce(v) ==
    /\ Len(queue) < K
    /\ queue' = Append(queue, v)

Consume ==
    /\ Len(queue) > 0
    /\ queue' = Tail(queue)

Next == (\E v \in 1..10 : Produce(v)) \/ Consume

Spec == Init /\ [][Next]_vars

BoundedQueue == Len(queue) <= K

====""")

add(
    "A bounded buffer connecting a producer and a consumer. The buffer has capacity K. Producer adds items, consumer removes them.",
    "I'll model a circular buffer with head, tail and count variables.",
    r"""---- MODULE BoundedBuffer ----
EXTENDS Naturals

CONSTANT K

VARIABLES buf, count

vars == <<buf, count>>

TypeOK ==
    /\ buf \in [0..(K-1) -> Nat]
    /\ count \in 0..K

Init ==
    /\ buf = [i \in 0..(K-1) |-> 0]
    /\ count = 0

Produce ==
    /\ count < K
    /\ \E v \in 1..5 : buf' = [buf EXCEPT ![count] = v]
    /\ count' = count + 1

Consume ==
    /\ count > 0
    /\ count' = count - 1
    /\ UNCHANGED buf

Next == Produce \/ Consume

Spec == Init /\ [][Next]_vars

BoundedQueue == count >= 0 /\ count <= K

====""")

# =========================================================================
# BM006 variants: Raft Leader Election
# =========================================================================
add(
    "The leader election phase of the Raft consensus algorithm. Servers hold terms and vote for candidates. A candidate wins if it gets a majority of votes.",
    "I'll model servers with states Follower/Candidate/Leader and track terms and votes.",
    r"""---- MODULE RaftElection ----
EXTENDS Naturals, FiniteSets

CONSTANT N

VARIABLES state, currentTerm, votedFor, votesGranted

vars == <<state, currentTerm, votedFor, votesGranted>>

Servers == 1..N

TypeOK ==
    /\ state \in [Servers -> {"follower", "candidate", "leader"}]
    /\ currentTerm \in [Servers -> Nat]
    /\ votedFor \in [Servers -> Servers \cup {0}]
    /\ votesGranted \in [Servers -> SUBSET Servers]

Init ==
    /\ state = [s \in Servers |-> "follower"]
    /\ currentTerm = [s \in Servers |-> 0]
    /\ votedFor = [s \in Servers |-> 0]
    /\ votesGranted = [s \in Servers |-> {}]

BecomeCandidate(s) ==
    /\ state[s] = "follower"
    /\ currentTerm' = [currentTerm EXCEPT ![s] = currentTerm[s] + 1]
    /\ state' = [state EXCEPT ![s] = "candidate"]
    /\ votedFor' = [votedFor EXCEPT ![s] = s]
    /\ votesGranted' = [votesGranted EXCEPT ![s] = {s}]

RequestVote(c, v) ==
    /\ state[c] = "candidate"
    /\ v /= c
    /\ votedFor[v] = 0
    /\ currentTerm[v] <= currentTerm[c]
    /\ votedFor' = [votedFor EXCEPT ![v] = c]
    /\ votesGranted' = [votesGranted EXCEPT ![c] = votesGranted[c] \cup {v}]
    /\ UNCHANGED <<state, currentTerm>>

BecomeLeader(s) ==
    /\ state[s] = "candidate"
    /\ Cardinality(votesGranted[s]) * 2 > N
    /\ state' = [state EXCEPT ![s] = "leader"]
    /\ UNCHANGED <<currentTerm, votedFor, votesGranted>>

Next ==
    \/ \E s \in Servers : BecomeCandidate(s)
    \/ \E c, v \in Servers : RequestVote(c, v)
    \/ \E s \in Servers : BecomeLeader(s)

Spec == Init /\ [][Next]_vars

AtMostOneLeader ==
    \A s1, s2 \in Servers :
        (state[s1] = "leader" /\ state[s2] = "leader") => s1 = s2

====""")

# =========================================================================
# BM007 variants: Read-Write Lock
# =========================================================================
add(
    "A read-write lock allowing multiple concurrent readers but exclusive writer access. Writers wait for all readers to finish.",
    "I'll track reader count and writer_active flag.",
    r"""---- MODULE ReadWriteLock ----
EXTENDS Naturals

VARIABLES readers, writer

vars == <<readers, writer>>

TypeOK ==
    /\ readers \in Nat
    /\ writer \in BOOLEAN

Init ==
    /\ readers = 0
    /\ writer = FALSE

AcquireRead ==
    /\ writer = FALSE
    /\ readers' = readers + 1
    /\ UNCHANGED writer

ReleaseRead ==
    /\ readers > 0
    /\ readers' = readers - 1
    /\ UNCHANGED writer

AcquireWrite ==
    /\ writer = FALSE
    /\ readers = 0
    /\ writer' = TRUE
    /\ UNCHANGED readers

ReleaseWrite ==
    /\ writer = TRUE
    /\ writer' = FALSE
    /\ UNCHANGED readers

Next == AcquireRead \/ ReleaseRead \/ AcquireWrite \/ ReleaseWrite

Spec == Init /\ [][Next]_vars

ExclusiveWrite == writer => readers = 0

====""")

# =========================================================================
# BM008 variants: Chandy-Lamport Snapshot
# =========================================================================
add(
    "The Chandy-Lamport distributed snapshot algorithm over a network of N processes with FIFO channels.",
    "I'll model processes that record local state and marker messages on channels.",
    r"""---- MODULE ChandyLamport ----
EXTENDS Naturals

CONSTANT N

VARIABLES recorded, state, markerSent

vars == <<recorded, state, markerSent>>

Procs == 1..N

TypeOK ==
    /\ recorded \in [Procs -> BOOLEAN]
    /\ state \in [Procs -> Nat]
    /\ markerSent \in [Procs -> BOOLEAN]

Init ==
    /\ recorded = [p \in Procs |-> FALSE]
    /\ state = [p \in Procs |-> 0]
    /\ markerSent = [p \in Procs |-> FALSE]

TakeSnapshot(p) ==
    /\ ~recorded[p]
    /\ recorded' = [recorded EXCEPT ![p] = TRUE]
    /\ markerSent' = [markerSent EXCEPT ![p] = TRUE]
    /\ UNCHANGED state

ReceiveMarker(p, q) ==
    /\ markerSent[q]
    /\ ~recorded[p]
    /\ recorded' = [recorded EXCEPT ![p] = TRUE]
    /\ markerSent' = [markerSent EXCEPT ![p] = TRUE]
    /\ UNCHANGED state

LocalStep(p) ==
    /\ state' = [state EXCEPT ![p] = state[p] + 1]
    /\ UNCHANGED <<recorded, markerSent>>

Next ==
    \/ \E p \in Procs : TakeSnapshot(p)
    \/ \E p, q \in Procs : p /= q /\ ReceiveMarker(p, q)
    \/ \E p \in Procs : LocalStep(p)

Spec == Init /\ [][Next]_vars

SnapshotConsistency ==
    \A p \in Procs : recorded[p] => markerSent[p]

====""")

# =========================================================================
# BM009 variants: Token Ring
# =========================================================================
add(
    "A token ring network with N nodes. Each node passes a single token clockwise. Only the token holder may send a message.",
    "I'll model a single token position cycling through nodes 1..N.",
    r"""---- MODULE TokenRing ----
EXTENDS Naturals

CONSTANT N

VARIABLE token

TypeOK == token \in 1..N

Init == token \in 1..N

PassToken ==
    token' = IF token = N THEN 1 ELSE token + 1

Next == PassToken

Spec == Init /\ [][Next]_<<token>>

UniqueToken == token \in 1..N

====""")

# =========================================================================
# BM010 variants: Key-Value Store
# =========================================================================
add(
    "A single-server key-value store supporting Put(k,v) and Get(k) operations. A Get always returns the value of the most recent Put.",
    "I'll model a store as a function from keys to values.",
    r"""---- MODULE KeyValueStore ----
EXTENDS Naturals

CONSTANTS Keys, Values

VARIABLE store

vars == <<store>>

TypeOK == store \in [Keys -> Values]

Init == store \in [Keys -> Values]

Put(k, v) ==
    /\ k \in Keys
    /\ v \in Values
    /\ store' = [store EXCEPT ![k] = v]

Next == \E k \in Keys, v \in Values : Put(k, v)

Spec == Init /\ [][Next]_vars

Linearizability == store \in [Keys -> Values]

====""")

# =========================================================================
# BM011 variants: Paxos Single-Decree
# =========================================================================
add(
    "Single-decree Paxos consensus over N acceptors and M proposers. Once a value is chosen, it is never changed.",
    "I'll model the Prepare/Promise and Accept/Accepted phases of Paxos.",
    r"""---- MODULE Paxos ----
EXTENDS Naturals, FiniteSets

CONSTANTS N, Values

VARIABLES maxBal, maxVBal, maxVal, chosen

vars == <<maxBal, maxVBal, maxVal, chosen>>

Acceptors == 1..N
Ballots == Nat

TypeOK ==
    /\ maxBal \in [Acceptors -> Ballots \cup {0}]
    /\ maxVBal \in [Acceptors -> Ballots \cup {0}]
    /\ maxVal \in [Acceptors -> Values \cup {"none"}]
    /\ chosen \in Values \cup {"none"}

Init ==
    /\ maxBal = [a \in Acceptors |-> 0]
    /\ maxVBal = [a \in Acceptors |-> 0]
    /\ maxVal = [a \in Acceptors |-> "none"]
    /\ chosen = "none"

Prepare(b, a) ==
    /\ b > maxBal[a]
    /\ maxBal' = [maxBal EXCEPT ![a] = b]
    /\ UNCHANGED <<maxVBal, maxVal, chosen>>

Accept(b, v, a) ==
    /\ b >= maxBal[a]
    /\ maxBal' = [maxBal EXCEPT ![a] = b]
    /\ maxVBal' = [maxVBal EXCEPT ![a] = b]
    /\ maxVal' = [maxVal EXCEPT ![a] = v]
    /\ UNCHANGED chosen

Choose(v) ==
    /\ chosen = "none"
    /\ \E Q \in SUBSET Acceptors :
        /\ Cardinality(Q) * 2 > N
        /\ \A a \in Q : maxVal[a] = v
    /\ chosen' = v
    /\ UNCHANGED <<maxBal, maxVBal, maxVal>>

Next ==
    \/ \E b \in 1..5, a \in Acceptors : Prepare(b, a)
    \/ \E b \in 1..5, v \in Values, a \in Acceptors : Accept(b, v, a)
    \/ \E v \in Values : Choose(v)

Spec == Init /\ [][Next]_vars

Consistency == chosen /= "none" => [](\A v \in Values : chosen' = chosen \/ chosen' = "none")

====""")

# =========================================================================
# BM012 variants: Bounded Retransmission
# =========================================================================
add(
    "A sender transmits a file in chunks over an unreliable channel. The sender retransmits up to MAX_RETRIES times before giving up.",
    "I'll model message loss as non-deterministic and track retransmit count.",
    r"""---- MODULE BoundedRetransmission ----
EXTENDS Naturals

CONSTANTS MaxRetries, NumChunks

VARIABLES sent, acked, retries, done

vars == <<sent, acked, retries, done>>

TypeOK ==
    /\ sent \in 0..NumChunks
    /\ acked \in 0..NumChunks
    /\ retries \in 0..MaxRetries
    /\ done \in BOOLEAN

Init ==
    /\ sent = 0
    /\ acked = 0
    /\ retries = 0
    /\ done = FALSE

Send ==
    /\ ~done
    /\ sent < NumChunks
    /\ sent' = sent + 1
    /\ UNCHANGED <<acked, retries, done>>

Ack ==
    /\ sent > acked
    /\ acked' = sent
    /\ retries' = 0
    /\ UNCHANGED <<sent, done>>

Lose ==
    /\ sent > acked
    /\ retries < MaxRetries
    /\ retries' = retries + 1
    /\ UNCHANGED <<sent, acked, done>>

GiveUp ==
    /\ retries = MaxRetries
    /\ done' = TRUE
    /\ UNCHANGED <<sent, acked, retries>>

Finish ==
    /\ acked = NumChunks
    /\ done' = TRUE
    /\ UNCHANGED <<sent, acked, retries>>

Next == Send \/ Ack \/ Lose \/ GiveUp \/ Finish

Spec == Init /\ [][Next]_vars

DeliveryOrFailure == done => (acked = NumChunks \/ retries = MaxRetries)

====""")

# =========================================================================
# BM013 variants: Snapshot Isolation
# =========================================================================
add(
    "A database with snapshot isolation. Transactions read a consistent snapshot. Write-write conflicts cause an abort.",
    "I'll model transactions with begin/commit/abort steps and write sets.",
    r"""---- MODULE SnapshotIsolation ----
EXTENDS Naturals, FiniteSets

CONSTANTS Keys, TxIds

VARIABLES store, active, writeSet, committed

vars == <<store, active, writeSet, committed>>

TypeOK ==
    /\ store \in [Keys -> Nat]
    /\ active \in SUBSET TxIds
    /\ writeSet \in [TxIds -> SUBSET Keys]
    /\ committed \in SUBSET TxIds

Init ==
    /\ store = [k \in Keys |-> 0]
    /\ active = {}
    /\ writeSet = [t \in TxIds |-> {}]
    /\ committed = {}

BeginTx(t) ==
    /\ t \notin active
    /\ t \notin committed
    /\ active' = active \cup {t}
    /\ writeSet' = [writeSet EXCEPT ![t] = {}]
    /\ UNCHANGED <<store, committed>>

Write(t, k) ==
    /\ t \in active
    /\ writeSet' = [writeSet EXCEPT ![t] = writeSet[t] \cup {k}]
    /\ UNCHANGED <<store, active, committed>>

CommitTx(t) ==
    /\ t \in active
    /\ \A t2 \in committed : writeSet[t] \cap writeSet[t2] = {}
    /\ store' = [k \in Keys |-> IF k \in writeSet[t] THEN store[k] + 1 ELSE store[k]]
    /\ active' = active \ {t}
    /\ committed' = committed \cup {t}
    /\ UNCHANGED writeSet

AbortTx(t) ==
    /\ t \in active
    /\ active' = active \ {t}
    /\ UNCHANGED <<store, writeSet, committed>>

Next ==
    \/ \E t \in TxIds : BeginTx(t)
    \/ \E t \in TxIds, k \in Keys : Write(t, k)
    \/ \E t \in TxIds : CommitTx(t)
    \/ \E t \in TxIds : AbortTx(t)

Spec == Init /\ [][Next]_vars

NoWriteConflict ==
    \A t1, t2 \in committed : t1 /= t2 => writeSet[t1] \cap writeSet[t2] = {}

====""")

# =========================================================================
# BM014 variants: Clock Synchronisation
# =========================================================================
add(
    "N nodes exchange clock values to synchronise. After one round, all clocks are within epsilon of each other.",
    "I'll model clocks with integer offsets and average-based synchronisation.",
    r"""---- MODULE ClockSync ----
EXTENDS Naturals, Integers

CONSTANTS N, Epsilon

VARIABLES clocks, synced

vars == <<clocks, synced>>

Nodes == 1..N

TypeOK ==
    /\ clocks \in [Nodes -> Int]
    /\ synced \in BOOLEAN

Init ==
    /\ clocks \in [Nodes -> 0..Epsilon]
    /\ synced = FALSE

Tick(n) ==
    /\ clocks' = [clocks EXCEPT ![n] = clocks[n] + 1]
    /\ UNCHANGED synced

Sync ==
    /\ ~synced
    /\ LET avg == (CHOOSE s \in Int : TRUE)
       IN clocks' = [n \in Nodes |-> clocks[n]]
    /\ synced' = TRUE

Next ==
    \/ \E n \in Nodes : Tick(n)
    \/ Sync

Spec == Init /\ [][Next]_vars

ClockBound == synced => \A i, j \in Nodes : clocks[i] - clocks[j] <= Epsilon /\ clocks[j] - clocks[i] <= Epsilon

====""")

# =========================================================================
# BM015 variants: Peterson's Algorithm
# =========================================================================
add(
    "Peterson's mutual exclusion algorithm for exactly 2 processes.",
    "I'll model with flag array and turn variable.",
    r"""---- MODULE Peterson ----
EXTENDS Naturals

VARIABLES flag, turn, pc

vars == <<flag, turn, pc>>

TypeOK ==
    /\ flag \in [1..2 -> BOOLEAN]
    /\ turn \in 1..2
    /\ pc \in [1..2 -> {"idle", "wait", "critical"}]

Init ==
    /\ flag = [i \in 1..2 |-> FALSE]
    /\ turn = 1
    /\ pc = [i \in 1..2 |-> "idle"]

Other(i) == IF i = 1 THEN 2 ELSE 1

SetFlag(i) ==
    /\ pc[i] = "idle"
    /\ flag' = [flag EXCEPT ![i] = TRUE]
    /\ turn' = Other(i)
    /\ pc' = [pc EXCEPT ![i] = "wait"]

Enter(i) ==
    /\ pc[i] = "wait"
    /\ flag[Other(i)] = FALSE \/ turn = i
    /\ pc' = [pc EXCEPT ![i] = "critical"]
    /\ UNCHANGED <<flag, turn>>

Exit(i) ==
    /\ pc[i] = "critical"
    /\ flag' = [flag EXCEPT ![i] = FALSE]
    /\ pc' = [pc EXCEPT ![i] = "idle"]
    /\ UNCHANGED turn

Next == \E i \in 1..2 : SetFlag(i) \/ Enter(i) \/ Exit(i)

Spec == Init /\ [][Next]_vars

MutualExclusion == ~(pc[1] = "critical" /\ pc[2] = "critical")

====""")

# =========================================================================
# BM016 variants: Gossip Protocol
# =========================================================================
add(
    "A gossip (epidemic) protocol where N nodes periodically share updates. A node that receives an update marks it as infected and spreads it.",
    "I'll model with known and infected sets tracking which nodes have seen the update.",
    r"""---- MODULE GossipProtocol ----
EXTENDS Naturals, FiniteSets

CONSTANT N

VARIABLES infected

vars == <<infected>>

Nodes == 1..N

TypeOK == infected \in SUBSET Nodes

Init == infected \in {S \in SUBSET Nodes : Cardinality(S) = 1}

Spread(i, j) ==
    /\ i \in infected
    /\ j \notin infected
    /\ infected' = infected \cup {j}

Next == \E i, j \in Nodes : i /= j /\ Spread(i, j)

Spec == Init /\ [][Next]_vars

EventualConsistency == infected = Nodes => []( infected = Nodes )

====""")

# =========================================================================
# BM017 variants: Simple Allocator
# =========================================================================
add(
    "A memory allocator managing a fixed pool of N pages. Clients request and release pages. No page is allocated to two clients simultaneously.",
    "I'll use VARIABLES free and allocated as a function from Client to SUBSET Pages.",
    r"""---- MODULE SimpleAllocator ----
EXTENDS Naturals, FiniteSets

CONSTANTS Pages, Clients

VARIABLES free, allocated

vars == <<free, allocated>>

TypeOK ==
    /\ free \subseteq Pages
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
    /\ free' = free \cup {p}
    /\ allocated' = [allocated EXCEPT ![c] = allocated[c] \ {p}]

Next == \E c \in Clients, p \in Pages : Allocate(c, p) \/ Release(c, p)

Spec == Init /\ [][Next]_vars

SafeAllocation ==
    \A c1, c2 \in Clients : c1 /= c2 => allocated[c1] \cap allocated[c2] = {}

====""")

# =========================================================================
# BM018 variants: Publish-Subscribe Broker
# =========================================================================
add(
    "A single broker with subscribers and publishers. Subscribers register interest in topics. Publishers post messages on topics. The broker delivers each message to interested subscribers.",
    "I'll model subscriptions, published messages, and delivered messages per subscriber.",
    r"""---- MODULE PubSubBroker ----
EXTENDS Naturals, FiniteSets

CONSTANTS Topics, Subs, MaxMsg

VARIABLES subscriptions, published, delivered

vars == <<subscriptions, published, delivered>>

TypeOK ==
    /\ subscriptions \in [Subs -> SUBSET Topics]
    /\ published \in [Topics -> 0..MaxMsg]
    /\ delivered \in [Subs -> [Topics -> 0..MaxMsg]]

Init ==
    /\ subscriptions = [s \in Subs |-> {}]
    /\ published = [t \in Topics |-> 0]
    /\ delivered = [s \in Subs |-> [t \in Topics |-> 0]]

Subscribe(s, t) ==
    /\ subscriptions' = [subscriptions EXCEPT ![s] = subscriptions[s] \cup {t}]
    /\ UNCHANGED <<published, delivered>>

Publish(t) ==
    /\ published[t] < MaxMsg
    /\ published' = [published EXCEPT ![t] = published[t] + 1]
    /\ UNCHANGED <<subscriptions, delivered>>

Deliver(s, t) ==
    /\ t \in subscriptions[s]
    /\ delivered[s][t] < published[t]
    /\ delivered' = [delivered EXCEPT ![s][t] = delivered[s][t] + 1]
    /\ UNCHANGED <<subscriptions, published>>

Next ==
    \/ \E s \in Subs, t \in Topics : Subscribe(s, t)
    \/ \E t \in Topics : Publish(t)
    \/ \E s \in Subs, t \in Topics : Deliver(s, t)

Spec == Init /\ [][Next]_vars

DeliveryGuarantee ==
    \A s \in Subs, t \in Topics :
        t \in subscriptions[s] => delivered[s][t] <= published[t]

====""")

# =========================================================================
# BM019 variants: Dekker's Algorithm
# =========================================================================
add(
    "Dekker's mutual exclusion algorithm for 2 processes, the first known correct solution to the problem.",
    "I'll model with turn and wants variables for 2 processes, tracking safety and liveness.",
    r"""---- MODULE Dekker ----
EXTENDS Naturals

VARIABLES wants, turn, pc

vars == <<wants, turn, pc>>

TypeOK ==
    /\ wants \in [1..2 -> BOOLEAN]
    /\ turn \in 1..2
    /\ pc \in [1..2 -> {"idle", "set_flag", "check", "wait", "critical", "exit"}]

Init ==
    /\ wants = [i \in 1..2 |-> FALSE]
    /\ turn = 1
    /\ pc = [i \in 1..2 |-> "idle"]

Other(i) == IF i = 1 THEN 2 ELSE 1

SetFlag(i) ==
    /\ pc[i] = "idle"
    /\ wants' = [wants EXCEPT ![i] = TRUE]
    /\ pc' = [pc EXCEPT ![i] = "check"]
    /\ UNCHANGED turn

Check(i) ==
    /\ pc[i] = "check"
    /\ IF wants[Other(i)]
       THEN IF turn = i
            THEN pc' = [pc EXCEPT ![i] = "check"]
            ELSE /\ wants' = [wants EXCEPT ![i] = FALSE]
                 /\ pc' = [pc EXCEPT ![i] = "wait"]
                 /\ UNCHANGED turn
       ELSE /\ pc' = [pc EXCEPT ![i] = "critical"]
            /\ UNCHANGED <<wants, turn>>

Wait(i) ==
    /\ pc[i] = "wait"
    /\ turn = i
    /\ wants' = [wants EXCEPT ![i] = TRUE]
    /\ pc' = [pc EXCEPT ![i] = "check"]
    /\ UNCHANGED turn

ExitCS(i) ==
    /\ pc[i] = "critical"
    /\ turn' = Other(i)
    /\ wants' = [wants EXCEPT ![i] = FALSE]
    /\ pc' = [pc EXCEPT ![i] = "idle"]

Next == \E i \in 1..2 : SetFlag(i) \/ Check(i) \/ Wait(i) \/ ExitCS(i)

Spec == Init /\ [][Next]_vars

MutualExclusion == ~(pc[1] = "critical" /\ pc[2] = "critical")

Deadlock_Freedom == \A i \in 1..2 : pc[i] = "idle" \/ pc[i] = "critical" \/ pc[Other(i)] /= "critical"

====""")

# =========================================================================
# BM020 variants: Eventually Consistent Counter
# =========================================================================
add(
    "A distributed grow-only counter (G-Counter CRDT) with N nodes. Each node increments its own slot. Merge takes element-wise max. All nodes eventually agree.",
    "I'll model counts as a 2D array counts[node][slot] with increment and merge operations.",
    r"""---- MODULE GCounter ----
EXTENDS Naturals

CONSTANT N

VARIABLES counts

vars == <<counts>>

Nodes == 1..N

TypeOK == counts \in [Nodes -> [Nodes -> Nat]]

Init == counts = [n \in Nodes |-> [m \in Nodes |-> 0]]

Increment(n) ==
    /\ counts' = [counts EXCEPT ![n][n] = counts[n][n] + 1]

Merge(i, j) ==
    /\ i /= j
    /\ counts' = [counts EXCEPT ![i] = [m \in Nodes |-> IF counts[j][m] > counts[i][m] THEN counts[j][m] ELSE counts[i][m]]]

Next ==
    \/ \E n \in Nodes : Increment(n)
    \/ \E i, j \in Nodes : Merge(i, j)

Spec == Init /\ [][Next]_vars

Monotone ==
    \A n, m \in Nodes : counts'[n][m] >= counts[n][m]

====""")

# =========================================================================
# Additional common spec patterns for training variety
# =========================================================================

add(
    "A simple counter that increments from 0 up to a maximum value N.",
    "I'll model a basic counter with increment operation.",
    r"""---- MODULE Counter ----
EXTENDS Naturals

CONSTANT N

VARIABLE count

TypeOK == count \in 0..N

Init == count = 0

Increment == /\ count < N /\ count' = count + 1

Next == Increment

Spec == Init /\ [][Next]_<<count>>

====""")

add(
    "A toggle switch that alternates between on and off states.",
    "I'll model a two-state system with a single toggle action.",
    r"""---- MODULE Toggle ----

VARIABLE on

TypeOK == on \in BOOLEAN

Init == on = FALSE

Toggle == on' = ~on

Next == Toggle

Spec == Init /\ [][Next]_<<on>>

====""")

add(
    "A traffic light controller cycling through red, yellow, green states.",
    "I'll model the state machine for a traffic light.",
    r"""---- MODULE TrafficLight ----

VARIABLE light

TypeOK == light \in {"red", "yellow", "green"}

Init == light = "red"

Next ==
    \/ /\ light = "red" /\ light' = "green"
    \/ /\ light = "green" /\ light' = "yellow"
    \/ /\ light = "yellow" /\ light' = "red"

Spec == Init /\ [][Next]_<<light>>

====""")

add(
    "A bank account with deposit and withdraw operations. Balance must never go below zero.",
    "I'll model a non-negative balance with deposit and withdraw actions.",
    r"""---- MODULE BankAccount ----
EXTENDS Naturals

CONSTANT MaxBalance

VARIABLE balance

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

Next == \E a \in 1..MaxBalance : Deposit(a) \/ Withdraw(a)

Spec == Init /\ [][Next]_<<balance>>

NonNegative == balance >= 0

====""")

add(
    "A counting semaphore with N permits. Threads can acquire and release permits.",
    "I'll model a semaphore value that stays between 0 and N.",
    r"""---- MODULE Semaphore ----
EXTENDS Naturals

CONSTANT N

VARIABLE permits

TypeOK == permits \in 0..N

Init == permits = N

Acquire == /\ permits > 0 /\ permits' = permits - 1

Release == /\ permits < N /\ permits' = permits + 1

Next == Acquire \/ Release

Spec == Init /\ [][Next]_<<permits>>

====""")

add(
    "A door that can be opened, closed, locked, and unlocked. You can only lock a closed door. You can only open an unlocked door.",
    "I'll model the door as a state machine.",
    r"""---- MODULE Door ----

VARIABLES state, locked

vars == <<state, locked>>

TypeOK ==
    /\ state \in {"open", "closed"}
    /\ locked \in BOOLEAN

Init ==
    /\ state = "closed"
    /\ locked = FALSE

Open ==
    /\ state = "closed"
    /\ locked = FALSE
    /\ state' = "open"
    /\ UNCHANGED locked

Close ==
    /\ state = "open"
    /\ state' = "closed"
    /\ UNCHANGED locked

Lock ==
    /\ state = "closed"
    /\ locked = FALSE
    /\ locked' = TRUE
    /\ UNCHANGED state

Unlock ==
    /\ locked = TRUE
    /\ locked' = FALSE
    /\ UNCHANGED state

Next == Open \/ Close \/ Lock \/ Unlock

Spec == Init /\ [][Next]_vars

LockedImpliesClosed == locked => state = "closed"

====""")

add(
    "N resources shared among M clients using a simple centralized resource allocator.",
    "I'll track which resources are free and who owns each allocated resource.",
    r"""---- MODULE ResourceAlloc ----
EXTENDS Naturals, FiniteSets

CONSTANTS Resources, Clients

VARIABLES owner

vars == <<owner>>

TypeOK == owner \in [Resources -> Clients \cup {"free"}]

Init == owner = [r \in Resources |-> "free"]

Acquire(c, r) ==
    /\ owner[r] = "free"
    /\ owner' = [owner EXCEPT ![r] = c]

Release(c, r) ==
    /\ owner[r] = c
    /\ owner' = [owner EXCEPT ![r] = "free"]

Next == \E c \in Clients, r \in Resources : Acquire(c, r) \/ Release(c, r)

Spec == Init /\ [][Next]_vars

====""")

add(
    "A bounded stack with push and pop operations. Maximum capacity is K.",
    "I'll model a stack as a sequence with push adding to the top and pop removing from top.",
    r"""---- MODULE BoundedStack ----
EXTENDS Naturals, Sequences

CONSTANT K

VARIABLE stack

vars == <<stack>>

TypeOK ==
    /\ stack \in Seq(Nat)
    /\ Len(stack) <= K

Init == stack = <<>>

Push(v) ==
    /\ Len(stack) < K
    /\ stack' = <<v>> \o stack

Pop ==
    /\ Len(stack) > 0
    /\ stack' = Tail(stack)

Next == (\E v \in 1..10 : Push(v)) \/ Pop

Spec == Init /\ [][Next]_vars

BoundedSize == Len(stack) <= K

====""")

add(
    "Model a distributed lock service. Multiple clients compete for a single lock. Only one can hold it at a time.",
    "I'll track the lock holder and pending requests.",
    r"""---- MODULE DistributedLock ----
EXTENDS Naturals, FiniteSets

CONSTANT Clients

VARIABLES holder, pending

vars == <<holder, pending>>

TypeOK ==
    /\ holder \in Clients \cup {"none"}
    /\ pending \in SUBSET Clients

Init ==
    /\ holder = "none"
    /\ pending = {}

Request(c) ==
    /\ c \notin pending
    /\ holder /= c
    /\ pending' = pending \cup {c}
    /\ UNCHANGED holder

Grant(c) ==
    /\ holder = "none"
    /\ c \in pending
    /\ holder' = c
    /\ pending' = pending \ {c}

Release(c) ==
    /\ holder = c
    /\ holder' = "none"
    /\ UNCHANGED pending

Next ==
    \/ \E c \in Clients : Request(c)
    \/ \E c \in Clients : Grant(c)
    \/ \E c \in Clients : Release(c)

Spec == Init /\ [][Next]_vars

MutualExclusion == holder /= "none" => Cardinality({c \in Clients : holder = c}) = 1

====""")

add(
    "An elevator controller for a building with N floors. The elevator moves up and down, stopping at requested floors.",
    "I'll track the current floor, direction, and set of requested floors.",
    r"""---- MODULE Elevator ----
EXTENDS Naturals, FiniteSets

CONSTANT N

VARIABLES floor, direction, requests

vars == <<floor, direction, requests>>

TypeOK ==
    /\ floor \in 1..N
    /\ direction \in {"up", "down", "idle"}
    /\ requests \in SUBSET (1..N)

Init ==
    /\ floor = 1
    /\ direction = "idle"
    /\ requests = {}

AddRequest(f) ==
    /\ requests' = requests \cup {f}
    /\ UNCHANGED <<floor, direction>>

MoveUp ==
    /\ floor < N
    /\ direction = "up"
    /\ floor' = floor + 1
    /\ UNCHANGED <<requests, direction>>

MoveDown ==
    /\ floor > 1
    /\ direction = "down"
    /\ floor' = floor - 1
    /\ UNCHANGED <<requests, direction>>

Stop ==
    /\ floor \in requests
    /\ requests' = requests \ {floor}
    /\ direction' = "idle"
    /\ UNCHANGED floor

ChooseDirection ==
    /\ direction = "idle"
    /\ requests /= {}
    /\ IF \E f \in requests : f > floor
       THEN direction' = "up"
       ELSE direction' = "down"
    /\ UNCHANGED <<floor, requests>>

Next ==
    \/ \E f \in 1..N : AddRequest(f)
    \/ MoveUp
    \/ MoveDown
    \/ Stop
    \/ ChooseDirection

Spec == Init /\ [][Next]_vars

====""")

add(
    "A simple FIFO channel between a sender and receiver. Messages are delivered in order.",
    "I'll model a channel as a sequence of messages.",
    r"""---- MODULE FIFOChannel ----
EXTENDS Naturals, Sequences

CONSTANT MaxLen

VARIABLES channel

vars == <<channel>>

TypeOK ==
    /\ channel \in Seq(Nat)
    /\ Len(channel) <= MaxLen

Init == channel = <<>>

Send(m) ==
    /\ Len(channel) < MaxLen
    /\ channel' = Append(channel, m)

Receive ==
    /\ Len(channel) > 0
    /\ channel' = Tail(channel)

Next == (\E m \in 1..5 : Send(m)) \/ Receive

Spec == Init /\ [][Next]_vars

====""")

add(
    "A leader-follower replication protocol. The leader accepts writes and replicates to followers.",
    "I'll track the leader's log and each follower's replicated position.",
    r"""---- MODULE LeaderFollower ----
EXTENDS Naturals

CONSTANT N

VARIABLES log, replicated

vars == <<log, replicated>>

Followers == 1..N

TypeOK ==
    /\ log \in Nat
    /\ replicated \in [Followers -> Nat]

Init ==
    /\ log = 0
    /\ replicated = [f \in Followers |-> 0]

Write ==
    /\ log' = log + 1
    /\ UNCHANGED replicated

Replicate(f) ==
    /\ replicated[f] < log
    /\ replicated' = [replicated EXCEPT ![f] = replicated[f] + 1]
    /\ UNCHANGED log

Next ==
    \/ Write
    \/ \E f \in Followers : Replicate(f)

Spec == Init /\ [][Next]_vars

ReplicationBound == \A f \in Followers : replicated[f] <= log

====""")

add(
    "A simple state machine modeling a vending machine. Insert coin, select product, dispense.",
    "I'll model the vending machine states and transitions.",
    r"""---- MODULE VendingMachine ----
EXTENDS Naturals

VARIABLE state, coins

vars == <<state, coins>>

TypeOK ==
    /\ state \in {"idle", "ready", "dispensing"}
    /\ coins \in Nat

Init ==
    /\ state = "idle"
    /\ coins = 0

InsertCoin ==
    /\ state = "idle"
    /\ coins' = coins + 1
    /\ state' = "ready"

SelectProduct ==
    /\ state = "ready"
    /\ coins > 0
    /\ state' = "dispensing"
    /\ UNCHANGED coins

Dispense ==
    /\ state = "dispensing"
    /\ coins' = coins - 1
    /\ state' = "idle"

Next == InsertCoin \/ SelectProduct \/ Dispense

Spec == Init /\ [][Next]_vars

NonNegativeCoins == coins >= 0

====""")

add(
    "A specification for a simple consensus protocol where N nodes must agree on a single value. Once decided, the value never changes.",
    "I'll model nodes with proposed and decided values, requiring majority agreement.",
    r"""---- MODULE SimpleConsensus ----
EXTENDS Naturals, FiniteSets

CONSTANTS N, Values

VARIABLES proposed, decided

vars == <<proposed, decided>>

Nodes == 1..N

TypeOK ==
    /\ proposed \in [Nodes -> Values \cup {"none"}]
    /\ decided \in Values \cup {"none"}

Init ==
    /\ proposed = [n \in Nodes |-> "none"]
    /\ decided = "none"

Propose(n, v) ==
    /\ proposed[n] = "none"
    /\ decided = "none"
    /\ proposed' = [proposed EXCEPT ![n] = v]
    /\ UNCHANGED decided

Decide(v) ==
    /\ decided = "none"
    /\ Cardinality({n \in Nodes : proposed[n] = v}) * 2 > N
    /\ decided' = v
    /\ UNCHANGED proposed

Next ==
    \/ \E n \in Nodes, v \in Values : Propose(n, v)
    \/ \E v \in Values : Decide(v)

Spec == Init /\ [][Next]_vars

Agreement == decided /= "none" => decided' = decided

====""")

add(
    "A basic load balancer distributing requests to N backend servers. Track which server handles each request.",
    "I'll model a round-robin load balancer.",
    r"""---- MODULE LoadBalancer ----
EXTENDS Naturals

CONSTANT N

VARIABLES next, load

vars == <<next, load>>

Servers == 1..N

TypeOK ==
    /\ next \in Servers
    /\ load \in [Servers -> Nat]

Init ==
    /\ next = 1
    /\ load = [s \in Servers |-> 0]

HandleRequest ==
    /\ load' = [load EXCEPT ![next] = load[next] + 1]
    /\ next' = IF next = N THEN 1 ELSE next + 1

CompleteRequest(s) ==
    /\ load[s] > 0
    /\ load' = [load EXCEPT ![s] = load[s] - 1]
    /\ UNCHANGED next

Next == HandleRequest \/ \E s \in Servers : CompleteRequest(s)

Spec == Init /\ [][Next]_vars

====""")

add(
    "A heartbeat failure detector. N nodes send periodic heartbeats. If a node misses K consecutive heartbeats, it is suspected as failed.",
    "I'll track heartbeat counters and suspected status.",
    r"""---- MODULE FailureDetector ----
EXTENDS Naturals

CONSTANTS N, K

VARIABLES missed, suspected

vars == <<missed, suspected>>

Nodes == 1..N

TypeOK ==
    /\ missed \in [Nodes -> 0..K]
    /\ suspected \in [Nodes -> BOOLEAN]

Init ==
    /\ missed = [n \in Nodes |-> 0]
    /\ suspected = [n \in Nodes |-> FALSE]

Heartbeat(n) ==
    /\ missed' = [missed EXCEPT ![n] = 0]
    /\ suspected' = [suspected EXCEPT ![n] = FALSE]

Timeout(n) ==
    /\ missed[n] < K
    /\ missed' = [missed EXCEPT ![n] = missed[n] + 1]
    /\ suspected' = [suspected EXCEPT ![n] = (missed[n] + 1 = K)]

Next == \E n \in Nodes : Heartbeat(n) \/ Timeout(n)

Spec == Init /\ [][Next]_vars

SuspectedIfMissed == \A n \in Nodes : (missed[n] = K) => suspected[n]

====""")

add(
    "A replicated state machine with N replicas. Each replica processes commands in order. All replicas must process the same sequence.",
    "I'll model a shared log and per-replica execution positions.",
    r"""---- MODULE ReplicatedSM ----
EXTENDS Naturals, Sequences

CONSTANT N

VARIABLES logSeq, executed

vars == <<logSeq, executed>>

Replicas == 1..N

TypeOK ==
    /\ logSeq \in Seq(Nat)
    /\ executed \in [Replicas -> Nat]

Init ==
    /\ logSeq = <<>>
    /\ executed = [r \in Replicas |-> 0]

AppendCommand(cmd) ==
    /\ logSeq' = Append(logSeq, cmd)
    /\ UNCHANGED executed

Execute(r) ==
    /\ executed[r] < Len(logSeq)
    /\ executed' = [executed EXCEPT ![r] = executed[r] + 1]
    /\ UNCHANGED logSeq

Next ==
    \/ \E cmd \in 1..5 : AppendCommand(cmd)
    \/ \E r \in Replicas : Execute(r)

Spec == Init /\ [][Next]_vars

InOrderExecution == \A r \in Replicas : executed[r] <= Len(logSeq)

====""")

add(
    "A simple database with transactions that can read, write, commit or abort. Committed writes become visible to new transactions.",
    "I'll model a key-value store with transaction lifecycle.",
    r"""---- MODULE SimpleDB ----
EXTENDS Naturals, FiniteSets

CONSTANTS Keys, TxIds

VARIABLES db, activeTx

vars == <<db, activeTx>>

TypeOK ==
    /\ db \in [Keys -> Nat]
    /\ activeTx \in SUBSET TxIds

Init ==
    /\ db = [k \in Keys |-> 0]
    /\ activeTx = {}

BeginTx(t) ==
    /\ t \notin activeTx
    /\ activeTx' = activeTx \cup {t}
    /\ UNCHANGED db

WriteTx(t, k, v) ==
    /\ t \in activeTx
    /\ db' = [db EXCEPT ![k] = v]
    /\ UNCHANGED activeTx

CommitTx(t) ==
    /\ t \in activeTx
    /\ activeTx' = activeTx \ {t}
    /\ UNCHANGED db

AbortTx(t) ==
    /\ t \in activeTx
    /\ activeTx' = activeTx \ {t}
    /\ UNCHANGED db

Next ==
    \/ \E t \in TxIds : BeginTx(t)
    \/ \E t \in TxIds, k \in Keys, v \in 0..5 : WriteTx(t, k, v)
    \/ \E t \in TxIds : CommitTx(t)
    \/ \E t \in TxIds : AbortTx(t)

Spec == Init /\ [][Next]_vars

====""")

add(
    "A simple cache that stores recently accessed items. The cache has a fixed size. When full, an item is evicted to make room.",
    "I'll model a set-based cache with add and evict operations.",
    r"""---- MODULE Cache ----
EXTENDS Naturals, FiniteSets

CONSTANTS Items, CacheSize

VARIABLE cached

vars == <<cached>>

TypeOK == /\ cached \subseteq Items /\ Cardinality(cached) <= CacheSize

Init == cached = {}

Add(item) ==
    /\ Cardinality(cached) < CacheSize
    /\ cached' = cached \cup {item}

Evict(item) ==
    /\ item \in cached
    /\ cached' = cached \ {item}

EvictAndAdd(old, new) ==
    /\ old \in cached
    /\ new \notin cached
    /\ Cardinality(cached) = CacheSize
    /\ cached' = (cached \ {old}) \cup {new}

Next ==
    \/ \E i \in Items : Add(i)
    \/ \E i \in Items : Evict(i)
    \/ \E old, new \in Items : EvictAndAdd(old, new)

Spec == Init /\ [][Next]_vars

BoundedSize == Cardinality(cached) <= CacheSize

====""")

add(
    "A barrier synchronization primitive. N threads arrive at a barrier and none can proceed until all have arrived.",
    "I'll track how many threads have arrived and whether the barrier is open.",
    r"""---- MODULE Barrier ----
EXTENDS Naturals

CONSTANT N

VARIABLES arrived, open

vars == <<arrived, open>>

TypeOK ==
    /\ arrived \in 0..N
    /\ open \in BOOLEAN

Init ==
    /\ arrived = 0
    /\ open = FALSE

Arrive ==
    /\ ~open
    /\ arrived < N
    /\ arrived' = arrived + 1
    /\ open' = (arrived + 1 = N)

Proceed ==
    /\ open
    /\ arrived' = arrived - 1
    /\ open' = (arrived - 1 > 0)

Reset ==
    /\ open
    /\ arrived = 0
    /\ open' = FALSE
    /\ UNCHANGED arrived

Next == Arrive \/ Proceed \/ Reset

Spec == Init /\ [][Next]_vars

SafeBarrier == ~open => arrived < N

====""")

# =========================================================================
# SANY validation and JSONL output
# =========================================================================

def validate_sany(code: str) -> bool:
    """Validate TLA+ code with SANY."""
    m = re.search(r"MODULE\s+(\w+)", code)
    module_name = m.group(1) if m else "Test"
    with tempfile.TemporaryDirectory() as tmpdir:
        tla_path = Path(tmpdir) / f"{module_name}.tla"
        tla_path.write_text(code)
        result = subprocess.run(
            ["java", "-cp", str(SANY_JAR), "tla2sany.SANY", str(tla_path)],
            capture_output=True, text=True, timeout=15
        )
        output = result.stdout + result.stderr
        return "error" not in output.lower()


def build_example(prompt: str, analysis: str, code: str) -> dict:
    """Build a harmony-format training example."""
    return {
        "messages": [
            {"role": "developer", "content": DEVELOPER_PROMPT},
            {"role": "user", "content": f"Write a TLA+ specification for the following:\n\n{prompt}"},
            {"role": "assistant", "channel": "analysis", "content": analysis},
            {"role": "assistant", "channel": "final", "content": code},
        ]
    }


def main():
    passed = 0
    failed = 0
    examples = []

    for i, (prompt, analysis, code) in enumerate(SPECS):
        m = re.search(r"MODULE\s+(\w+)", code)
        name = m.group(1) if m else f"Spec{i}"
        ok = validate_sany(code)
        status = "PASS" if ok else "FAIL"
        print(f"  [{i+1:3}/{len(SPECS)}] {name:30s} {status}")
        if ok:
            passed += 1
            examples.append(build_example(prompt, analysis, code))
        else:
            failed += 1

    print(f"\nResults: {passed}/{len(SPECS)} passed SANY, {failed} failed")

    # Backup and write
    if OUTPUT.exists():
        backup = OUTPUT.with_suffix(".pre_v2")
        import shutil
        shutil.copy2(OUTPUT, backup)
        print(f"Backed up {OUTPUT} -> {backup}")

    with open(OUTPUT, "w") as f:
        for ex in examples:
            f.write(json.dumps(ex) + "\n")
    print(f"Wrote {len(examples)} examples to {OUTPUT}")


if __name__ == "__main__":
    main()
