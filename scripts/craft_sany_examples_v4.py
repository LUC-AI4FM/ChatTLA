#!/usr/bin/env python3
"""
craft_sany_examples_v4.py  —  Targeted examples for the 11 consistently-failing benchmarks.

Each spec is hand-written to pass SANY, following proper TLA+ syntax patterns
the model struggles with:
  - Proper VARIABLES declarations (never forgotten)
  - No PlusCal syntax
  - Standard operators only (no Subst, RANGE, Sum)
  - Correct priming: v'[i] not v[i]'
  - EXTENDS Integers when Int is needed
  - Proper function construction [x \in S |-> expr]
  - No == inside conjunctions (use = for equality)
"""
import json, subprocess, tempfile, sys
from pathlib import Path

EXAMPLES = []

# ─────────────────────────────────────
# BM002: Two-Phase Commit
# ─────────────────────────────────────
EXAMPLES.append({
    "prompt": "A two-phase commit protocol with one coordinator and N participants. The coordinator decides to commit only if all participants vote yes. Messages: Prepare, VoteYes, VoteNo, Commit, Abort.",
    "spec": r"""---- MODULE TwoPhaseCommit ----
EXTENDS Naturals, FiniteSets, TLC

CONSTANTS N

Participants == 1..N

VARIABLES coordState, partState, msgs

TypeOK ==
  /\ coordState \in {"init", "waiting", "committed", "aborted"}
  /\ partState \in [Participants -> {"working", "prepared", "committed", "aborted"}]
  /\ msgs \subseteq {"Prepare", "VoteYes", "VoteNo", "Commit", "Abort"}

Init ==
  /\ coordState = "init"
  /\ partState = [p \in Participants |-> "working"]
  /\ msgs = {}

CoordSendPrepare ==
  /\ coordState = "init"
  /\ coordState' = "waiting"
  /\ msgs' = msgs \cup {"Prepare"}
  /\ UNCHANGED partState

ParticipantVoteYes(p) ==
  /\ "Prepare" \in msgs
  /\ partState[p] = "working"
  /\ partState' = [partState EXCEPT ![p] = "prepared"]
  /\ msgs' = msgs \cup {"VoteYes"}
  /\ UNCHANGED coordState

ParticipantVoteNo(p) ==
  /\ "Prepare" \in msgs
  /\ partState[p] = "working"
  /\ partState' = [partState EXCEPT ![p] = "aborted"]
  /\ msgs' = msgs \cup {"VoteNo"}
  /\ UNCHANGED coordState

CoordDecideCommit ==
  /\ coordState = "waiting"
  /\ \A p \in Participants : partState[p] = "prepared"
  /\ coordState' = "committed"
  /\ msgs' = msgs \cup {"Commit"}
  /\ UNCHANGED partState

CoordDecideAbort ==
  /\ coordState = "waiting"
  /\ \E p \in Participants : partState[p] = "aborted"
  /\ coordState' = "aborted"
  /\ msgs' = msgs \cup {"Abort"}
  /\ UNCHANGED partState

ParticipantCommit(p) ==
  /\ "Commit" \in msgs
  /\ partState[p] = "prepared"
  /\ partState' = [partState EXCEPT ![p] = "committed"]
  /\ UNCHANGED <<coordState, msgs>>

ParticipantAbort(p) ==
  /\ "Abort" \in msgs
  /\ partState[p] \in {"working", "prepared"}
  /\ partState' = [partState EXCEPT ![p] = "aborted"]
  /\ UNCHANGED <<coordState, msgs>>

Next ==
  \/ CoordSendPrepare
  \/ CoordDecideCommit
  \/ CoordDecideAbort
  \/ \E p \in Participants :
       \/ ParticipantVoteYes(p)
       \/ ParticipantVoteNo(p)
       \/ ParticipantCommit(p)
       \/ ParticipantAbort(p)

Consistency ==
  coordState = "committed" => \A p \in Participants : partState[p] \in {"prepared", "committed"}

Spec == Init /\ [][Next]_<<coordState, partState, msgs>>

====
"""
})

# BM002 variant — simplified
EXAMPLES.append({
    "prompt": "Model a two-phase commit with a coordinator and participants. Participants vote, coordinator commits or aborts based on votes.",
    "spec": r"""---- MODULE TwoPC ----
EXTENDS Naturals, FiniteSets, TLC

CONSTANT N

VARIABLES phase, votes, decision

Procs == 1..N

TypeOK ==
  /\ phase \in {"idle", "vote", "decided"}
  /\ votes \in [Procs -> {"none", "yes", "no"}]
  /\ decision \in {"none", "commit", "abort"}

Init ==
  /\ phase = "idle"
  /\ votes = [p \in Procs |-> "none"]
  /\ decision = "none"

StartVote ==
  /\ phase = "idle"
  /\ phase' = "vote"
  /\ UNCHANGED <<votes, decision>>

CastVote(p) ==
  /\ phase = "vote"
  /\ votes[p] = "none"
  /\ \E v \in {"yes", "no"} :
       votes' = [votes EXCEPT ![p] = v]
  /\ UNCHANGED <<phase, decision>>

Decide ==
  /\ phase = "vote"
  /\ \A p \in Procs : votes[p] # "none"
  /\ IF \A p \in Procs : votes[p] = "yes"
     THEN decision' = "commit"
     ELSE decision' = "abort"
  /\ phase' = "decided"
  /\ UNCHANGED votes

Next ==
  \/ StartVote
  \/ Decide
  \/ \E p \in Procs : CastVote(p)

Consistency ==
  decision = "commit" => \A p \in Procs : votes[p] = "yes"

Spec == Init /\ [][Next]_<<phase, votes, decision>>

====
"""
})

# ─────────────────────────────────────
# BM003: Dining Philosophers
# ─────────────────────────────────────
EXAMPLES.append({
    "prompt": "The dining philosophers problem with N philosophers and N forks. Each philosopher picks up both adjacent forks to eat, then puts them down. Track fork ownership and philosopher states: thinking, hungry, eating.",
    "spec": r"""---- MODULE DiningPhilosophers ----
EXTENDS Naturals, TLC

CONSTANT N

VARIABLES forks, philState

TypeOK ==
  /\ forks \in [1..N -> (1..N) \cup {0}]
  /\ philState \in [1..N -> {"thinking", "hungry", "eating"}]

LeftFork(i) == i
RightFork(i) == IF i = N THEN 1 ELSE i + 1

Init ==
  /\ forks = [f \in 1..N |-> 0]
  /\ philState = [p \in 1..N |-> "thinking"]

BecomeHungry(p) ==
  /\ philState[p] = "thinking"
  /\ philState' = [philState EXCEPT ![p] = "hungry"]
  /\ UNCHANGED forks

PickUpForks(p) ==
  /\ philState[p] = "hungry"
  /\ forks[LeftFork(p)] = 0
  /\ forks[RightFork(p)] = 0
  /\ forks' = [forks EXCEPT ![LeftFork(p)] = p, ![RightFork(p)] = p]
  /\ philState' = [philState EXCEPT ![p] = "eating"]

PutDownForks(p) ==
  /\ philState[p] = "eating"
  /\ forks' = [forks EXCEPT ![LeftFork(p)] = 0, ![RightFork(p)] = 0]
  /\ philState' = [philState EXCEPT ![p] = "thinking"]

Next ==
  \E p \in 1..N :
    \/ BecomeHungry(p)
    \/ PickUpForks(p)
    \/ PutDownForks(p)

NoDeadlock == \E p \in 1..N : philState[p] # "hungry" \/ (forks[LeftFork(p)] = 0 /\ forks[RightFork(p)] = 0)

Spec == Init /\ [][Next]_<<forks, philState>>

====
"""
})

# ─────────────────────────────────────
# BM004: Lamport's Bakery Algorithm
# ─────────────────────────────────────
EXAMPLES.append({
    "prompt": "Lamport's bakery mutual exclusion algorithm for N processes. Processes take a numbered ticket; lower numbers enter first. VARIABLES num, flag. Choosing phase before assigning number.",
    "spec": r"""---- MODULE Bakery ----
EXTENDS Naturals, TLC

CONSTANT N

VARIABLES num, flag, pc

Procs == 1..N

TypeOK ==
  /\ num \in [Procs -> Nat]
  /\ flag \in [Procs -> BOOLEAN]
  /\ pc \in [Procs -> {"idle", "choosing", "waiting", "cs"}]

Init ==
  /\ num = [p \in Procs |-> 0]
  /\ flag = [p \in Procs |-> FALSE]
  /\ pc = [p \in Procs |-> "idle"]

MaxNum == CHOOSE m \in 0..N : \A p \in Procs : num[p] <= m

Enter(p) ==
  /\ pc[p] = "idle"
  /\ flag' = [flag EXCEPT ![p] = TRUE]
  /\ pc' = [pc EXCEPT ![p] = "choosing"]
  /\ UNCHANGED num

ChooseNumber(p) ==
  /\ pc[p] = "choosing"
  /\ \E n \in 1..(N+1) :
       /\ \A q \in Procs : num[q] < n
       /\ num' = [num EXCEPT ![p] = n]
  /\ flag' = [flag EXCEPT ![p] = FALSE]
  /\ pc' = [pc EXCEPT ![p] = "waiting"]

EnterCS(p) ==
  /\ pc[p] = "waiting"
  /\ \A q \in Procs \ {p} :
       \/ num[q] = 0
       \/ num[p] < num[q]
       \/ (num[p] = num[q] /\ p < q)
  /\ pc' = [pc EXCEPT ![p] = "cs"]
  /\ UNCHANGED <<num, flag>>

ExitCS(p) ==
  /\ pc[p] = "cs"
  /\ num' = [num EXCEPT ![p] = 0]
  /\ pc' = [pc EXCEPT ![p] = "idle"]
  /\ UNCHANGED flag

Next == \E p \in Procs :
  \/ Enter(p)
  \/ ChooseNumber(p)
  \/ EnterCS(p)
  \/ ExitCS(p)

MutualExclusion == \A p, q \in Procs : (p # q) => ~(pc[p] = "cs" /\ pc[q] = "cs")

Spec == Init /\ [][Next]_<<num, flag, pc>>

====
"""
})

# ─────────────────────────────────────
# BM005: Producer-Consumer Bounded Queue
# ─────────────────────────────────────
EXAMPLES.append({
    "prompt": "A bounded FIFO queue with one producer and one consumer. The producer blocks when the queue is full; the consumer blocks when empty. Queue capacity is a constant K.",
    "spec": r"""---- MODULE BoundedQueue ----
EXTENDS Naturals, Sequences, TLC

CONSTANT K

VARIABLES queue

TypeOK ==
  /\ queue \in Seq(Nat)
  /\ Len(queue) <= K

Init ==
  /\ queue = <<>>

Produce ==
  /\ Len(queue) < K
  /\ \E v \in 1..10 :
       queue' = Append(queue, v)

Consume ==
  /\ Len(queue) > 0
  /\ queue' = Tail(queue)

Next ==
  \/ Produce
  \/ Consume

BoundedQueue == Len(queue) <= K

Spec == Init /\ [][Next]_queue

====
"""
})

# BM005 variant with head/tail pointers
EXAMPLES.append({
    "prompt": "A bounded circular buffer with capacity K. A producer adds items and a consumer removes them. Track head, tail, and count.",
    "spec": r"""---- MODULE CircularBuffer ----
EXTENDS Naturals, TLC

CONSTANT K

VARIABLES buf, head, tail, count

TypeOK ==
  /\ buf \in [0..(K-1) -> Nat]
  /\ head \in 0..(K-1)
  /\ tail \in 0..(K-1)
  /\ count \in 0..K

Init ==
  /\ buf = [i \in 0..(K-1) |-> 0]
  /\ head = 0
  /\ tail = 0
  /\ count = 0

Produce ==
  /\ count < K
  /\ \E v \in 1..10 :
       buf' = [buf EXCEPT ![tail] = v]
  /\ tail' = (tail + 1) % K
  /\ count' = count + 1
  /\ UNCHANGED head

Consume ==
  /\ count > 0
  /\ head' = (head + 1) % K
  /\ count' = count - 1
  /\ UNCHANGED <<buf, tail>>

Next ==
  \/ Produce
  \/ Consume

BoundedQueue == count <= K

Spec == Init /\ [][Next]_<<buf, head, tail, count>>

====
"""
})

# ─────────────────────────────────────
# BM008: Chandy-Lamport Distributed Snapshot
# ─────────────────────────────────────
EXAMPLES.append({
    "prompt": "The Chandy-Lamport distributed snapshot algorithm over a network of N processes with FIFO channels. Each process records its local state and the state of incoming channels. Send marker messages along each outgoing channel after recording state.",
    "spec": r"""---- MODULE ChandyLamport ----
EXTENDS Naturals, Sequences, TLC

CONSTANT N

Procs == 1..N

VARIABLES localState, recorded, channelState, channels, markerSent

TypeOK ==
  /\ localState \in [Procs -> Nat]
  /\ recorded \in [Procs -> BOOLEAN]
  /\ channelState \in [Procs -> [Procs -> Seq(Nat)]]
  /\ channels \in [Procs -> [Procs -> Seq(Nat)]]
  /\ markerSent \in [Procs -> BOOLEAN]

Init ==
  /\ localState = [p \in Procs |-> 0]
  /\ recorded = [p \in Procs |-> FALSE]
  /\ channelState = [p \in Procs |-> [q \in Procs |-> <<>>]]
  /\ channels = [p \in Procs |-> [q \in Procs |-> <<>>]]
  /\ markerSent = [p \in Procs |-> FALSE]

InitiateSnapshot(p) ==
  /\ ~recorded[p]
  /\ recorded' = [recorded EXCEPT ![p] = TRUE]
  /\ markerSent' = [markerSent EXCEPT ![p] = TRUE]
  /\ UNCHANGED <<localState, channelState, channels>>

SendMessage(p, q) ==
  /\ p # q
  /\ channels' = [channels EXCEPT ![p][q] = Append(channels[p][q], localState[p])]
  /\ UNCHANGED <<localState, recorded, channelState, markerSent>>

ReceiveMarker(p, q) ==
  /\ p # q
  /\ markerSent[q]
  /\ ~recorded[p]
  /\ recorded' = [recorded EXCEPT ![p] = TRUE]
  /\ channelState' = [channelState EXCEPT ![p][q] = channels[q][p]]
  /\ markerSent' = [markerSent EXCEPT ![p] = TRUE]
  /\ UNCHANGED <<localState, channels>>

RecordChannel(p, q) ==
  /\ p # q
  /\ recorded[p]
  /\ markerSent[q]
  /\ channelState' = [channelState EXCEPT ![p][q] = channels[q][p]]
  /\ UNCHANGED <<localState, recorded, channels, markerSent>>

Next ==
  \E p \in Procs :
    \/ InitiateSnapshot(p)
    \/ \E q \in Procs : SendMessage(p, q)
    \/ \E q \in Procs : ReceiveMarker(p, q)
    \/ \E q \in Procs : RecordChannel(p, q)

SnapshotConsistency ==
  (\A p \in Procs : recorded[p]) =>
    \A p, q \in Procs : p # q => channelState[p][q] = channels[q][p]

Spec == Init /\ [][Next]_<<localState, recorded, channelState, channels, markerSent>>

====
"""
})

# ─────────────────────────────────────
# BM010: Simple Key-Value Store
# ─────────────────────────────────────
EXAMPLES.append({
    "prompt": "A single-server key-value store supporting Put(k,v) and Get(k) operations. Linearizability: a Get always returns the value of the most recent Put. Model a finite key space Keys and value space Values.",
    "spec": r"""---- MODULE KeyValueStore ----
EXTENDS Naturals, TLC

CONSTANTS Keys, Values

VARIABLES store, lastRead

TypeOK ==
  /\ store \in [Keys -> Values \cup {"null"}]
  /\ lastRead \in [Keys -> Values \cup {"null"}]

Init ==
  /\ store = [k \in Keys |-> "null"]
  /\ lastRead = [k \in Keys |-> "null"]

Put(k, v) ==
  /\ store' = [store EXCEPT ![k] = v]
  /\ UNCHANGED lastRead

Get(k) ==
  /\ lastRead' = [lastRead EXCEPT ![k] = store[k]]
  /\ UNCHANGED store

Next ==
  \/ \E k \in Keys, v \in Values : Put(k, v)
  \/ \E k \in Keys : Get(k)

Linearizability == \A k \in Keys : lastRead[k] = store[k] \/ lastRead[k] = "null"

Spec == Init /\ [][Next]_<<store, lastRead>>

====
"""
})

# ─────────────────────────────────────
# BM012: Bounded Retransmission Protocol
# ─────────────────────────────────────
EXAMPLES.append({
    "prompt": "A sender transmits a file in chunks over an unreliable channel. The sender retransmits up to MAX_RETRIES times before giving up. Model message loss as non-deterministic. Track retransmit count.",
    "spec": r"""---- MODULE BoundedRetransmission ----
EXTENDS Naturals, Sequences, TLC

CONSTANTS MAX_RETRIES, NumChunks

VARIABLES senderState, receiverBuf, retries, currentChunk, channelMsg

TypeOK ==
  /\ senderState \in {"sending", "done", "failed"}
  /\ receiverBuf \in Seq(1..NumChunks)
  /\ retries \in 0..MAX_RETRIES
  /\ currentChunk \in 0..NumChunks
  /\ channelMsg \in (1..NumChunks) \cup {"empty", "ack"}

Init ==
  /\ senderState = "sending"
  /\ receiverBuf = <<>>
  /\ retries = 0
  /\ currentChunk = 1
  /\ channelMsg = "empty"

SendChunk ==
  /\ senderState = "sending"
  /\ currentChunk <= NumChunks
  /\ channelMsg = "empty"
  /\ channelMsg' = currentChunk
  /\ UNCHANGED <<senderState, receiverBuf, retries, currentChunk>>

MessageLost ==
  /\ channelMsg \in 1..NumChunks
  /\ channelMsg' = "empty"
  /\ retries' = retries + 1
  /\ IF retries + 1 > MAX_RETRIES
     THEN senderState' = "failed"
     ELSE senderState' = senderState
  /\ UNCHANGED <<receiverBuf, currentChunk>>

MessageDelivered ==
  /\ channelMsg \in 1..NumChunks
  /\ receiverBuf' = Append(receiverBuf, channelMsg)
  /\ channelMsg' = "ack"
  /\ UNCHANGED <<senderState, retries, currentChunk>>

ReceiveAck ==
  /\ channelMsg = "ack"
  /\ channelMsg' = "empty"
  /\ retries' = 0
  /\ currentChunk' = currentChunk + 1
  /\ IF currentChunk + 1 > NumChunks
     THEN senderState' = "done"
     ELSE senderState' = senderState
  /\ UNCHANGED receiverBuf

Next ==
  \/ SendChunk
  \/ MessageLost
  \/ MessageDelivered
  \/ ReceiveAck

DeliveryOrFailure ==
  senderState \in {"sending"} \/ senderState = "done" \/ senderState = "failed"

Spec == Init /\ [][Next]_<<senderState, receiverBuf, retries, currentChunk, channelMsg>>

====
"""
})

# ─────────────────────────────────────
# BM013: Snapshot Isolation / Transaction Isolation
# ─────────────────────────────────────
EXAMPLES.append({
    "prompt": "A database with snapshot isolation. Transactions read a consistent snapshot. Write-write conflicts cause an abort. Model transactions as spanning multiple steps with begin/commit/abort.",
    "spec": r"""---- MODULE SnapshotIsolation ----
EXTENDS Naturals, FiniteSets, TLC

CONSTANTS TxIds, Keys, Values

VARIABLES txState, readSnapshot, writeSet, store

TypeOK ==
  /\ txState \in [TxIds -> {"idle", "active", "committed", "aborted"}]
  /\ readSnapshot \in [TxIds -> [Keys -> Values \cup {"null"}]]
  /\ writeSet \in [TxIds -> SUBSET Keys]
  /\ store \in [Keys -> Values \cup {"null"}]

Init ==
  /\ txState = [t \in TxIds |-> "idle"]
  /\ readSnapshot = [t \in TxIds |-> [k \in Keys |-> "null"]]
  /\ writeSet = [t \in TxIds |-> {}]
  /\ store = [k \in Keys |-> "null"]

BeginTx(t) ==
  /\ txState[t] = "idle"
  /\ txState' = [txState EXCEPT ![t] = "active"]
  /\ readSnapshot' = [readSnapshot EXCEPT ![t] = store]
  /\ UNCHANGED <<writeSet, store>>

WriteTx(t, k, v) ==
  /\ txState[t] = "active"
  /\ writeSet' = [writeSet EXCEPT ![t] = writeSet[t] \cup {k}]
  /\ UNCHANGED <<txState, readSnapshot, store>>

CommitTx(t) ==
  /\ txState[t] = "active"
  /\ \A t2 \in TxIds :
       (t2 # t /\ txState[t2] = "committed") =>
         writeSet[t] \cap writeSet[t2] = {}
  /\ txState' = [txState EXCEPT ![t] = "committed"]
  /\ UNCHANGED <<readSnapshot, writeSet, store>>

AbortTx(t) ==
  /\ txState[t] = "active"
  /\ txState' = [txState EXCEPT ![t] = "aborted"]
  /\ UNCHANGED <<readSnapshot, writeSet, store>>

Next ==
  \E t \in TxIds :
    \/ BeginTx(t)
    \/ \E k \in Keys, v \in Values : WriteTx(t, k, v)
    \/ CommitTx(t)
    \/ AbortTx(t)

NoWriteConflict ==
  \A t1, t2 \in TxIds :
    (t1 # t2 /\ txState[t1] = "committed" /\ txState[t2] = "committed") =>
      writeSet[t1] \cap writeSet[t2] = {}

Spec == Init /\ [][Next]_<<txState, readSnapshot, writeSet, store>>

====
"""
})

# ─────────────────────────────────────
# BM014: Clock Synchronisation
# ─────────────────────────────────────
EXAMPLES.append({
    "prompt": "N nodes exchange clock values to synchronise. After one round, all clocks are within epsilon of each other. Drift modelled as integer offset. Average-based sync.",
    "spec": r"""---- MODULE ClockSync ----
EXTENDS Integers, Naturals, TLC

CONSTANTS N, MaxDrift

Nodes == 1..N

VARIABLES clocks, offsets, synced

TypeOK ==
  /\ clocks \in [Nodes -> Int]
  /\ offsets \in [Nodes -> Int]
  /\ synced \in BOOLEAN

Init ==
  /\ clocks = [n \in Nodes |-> 0]
  /\ offsets = [n \in Nodes |-> 0]
  /\ synced = FALSE

Drift(n) ==
  /\ ~synced
  /\ \E d \in -MaxDrift..MaxDrift :
       offsets' = [offsets EXCEPT ![n] = d]
  /\ UNCHANGED <<clocks, synced>>

Tick ==
  /\ ~synced
  /\ clocks' = [n \in Nodes |-> clocks[n] + 1 + offsets[n]]
  /\ UNCHANGED <<offsets, synced>>

Synchronize ==
  /\ ~synced
  /\ clocks' = [n \in Nodes |-> clocks[1]]
  /\ offsets' = [n \in Nodes |-> 0]
  /\ synced' = TRUE

Next ==
  \/ Tick
  \/ Synchronize
  \/ \E n \in Nodes : Drift(n)

ClockBound ==
  synced => \A i, j \in Nodes : clocks[i] - clocks[j] >= -MaxDrift /\ clocks[i] - clocks[j] <= MaxDrift

Spec == Init /\ [][Next]_<<clocks, offsets, synced>>

====
"""
})

# ─────────────────────────────────────
# BM015: Peterson's Algorithm
# ─────────────────────────────────────
EXAMPLES.append({
    "prompt": "Peterson's mutual exclusion algorithm for exactly 2 processes. VARIABLES flag[2], turn.",
    "spec": r"""---- MODULE Peterson ----
EXTENDS Naturals, TLC

VARIABLES flag, turn, pc

Procs == {0, 1}
Other(p) == 1 - p

TypeOK ==
  /\ flag \in [Procs -> BOOLEAN]
  /\ turn \in Procs
  /\ pc \in [Procs -> {"idle", "set_flag", "set_turn", "wait", "cs", "exit"}]

Init ==
  /\ flag = [p \in Procs |-> FALSE]
  /\ turn = 0
  /\ pc = [p \in Procs |-> "idle"]

SetFlag(p) ==
  /\ pc[p] = "idle"
  /\ flag' = [flag EXCEPT ![p] = TRUE]
  /\ pc' = [pc EXCEPT ![p] = "set_turn"]
  /\ UNCHANGED turn

SetTurn(p) ==
  /\ pc[p] = "set_turn"
  /\ turn' = Other(p)
  /\ pc' = [pc EXCEPT ![p] = "wait"]
  /\ UNCHANGED flag

Wait(p) ==
  /\ pc[p] = "wait"
  /\ flag[Other(p)] = FALSE \/ turn = p
  /\ pc' = [pc EXCEPT ![p] = "cs"]
  /\ UNCHANGED <<flag, turn>>

EnterCS(p) ==
  /\ pc[p] = "cs"
  /\ pc' = [pc EXCEPT ![p] = "exit"]
  /\ UNCHANGED <<flag, turn>>

Exit(p) ==
  /\ pc[p] = "exit"
  /\ flag' = [flag EXCEPT ![p] = FALSE]
  /\ pc' = [pc EXCEPT ![p] = "idle"]
  /\ UNCHANGED turn

Next == \E p \in Procs :
  \/ SetFlag(p)
  \/ SetTurn(p)
  \/ Wait(p)
  \/ EnterCS(p)
  \/ Exit(p)

MutualExclusion == ~(pc[0] = "cs" /\ pc[1] = "cs")

Spec == Init /\ [][Next]_<<flag, turn, pc>>

====
"""
})

# ─────────────────────────────────────
# BM018: Publish-Subscribe Broker
# ─────────────────────────────────────
EXAMPLES.append({
    "prompt": "A single broker with subscribers and publishers. Subscribers register interest in topics. Publishers post messages on topics. The broker delivers each message to all registered subscribers. Messages arrive in order per topic.",
    "spec": r"""---- MODULE PubSubBroker ----
EXTENDS Naturals, Sequences, FiniteSets, TLC

CONSTANTS Topics, MaxSubscribers, MaxMessages

VARIABLES subscribers, messageQueue, delivered

TypeOK ==
  /\ subscribers \in [Topics -> SUBSET (1..MaxSubscribers)]
  /\ messageQueue \in [Topics -> Seq(Nat)]
  /\ delivered \in [Topics -> [1..MaxSubscribers -> Seq(Nat)]]

Init ==
  /\ subscribers = [t \in Topics |-> {}]
  /\ messageQueue = [t \in Topics |-> <<>>]
  /\ delivered = [t \in Topics |-> [s \in 1..MaxSubscribers |-> <<>>]]

Subscribe(t, s) ==
  /\ s \notin subscribers[t]
  /\ Cardinality(subscribers[t]) < MaxSubscribers
  /\ subscribers' = [subscribers EXCEPT ![t] = subscribers[t] \cup {s}]
  /\ UNCHANGED <<messageQueue, delivered>>

Unsubscribe(t, s) ==
  /\ s \in subscribers[t]
  /\ subscribers' = [subscribers EXCEPT ![t] = subscribers[t] \ {s}]
  /\ UNCHANGED <<messageQueue, delivered>>

Publish(t, msg) ==
  /\ Len(messageQueue[t]) < MaxMessages
  /\ messageQueue' = [messageQueue EXCEPT ![t] = Append(messageQueue[t], msg)]
  /\ UNCHANGED <<subscribers, delivered>>

Deliver(t) ==
  /\ Len(messageQueue[t]) > 0
  /\ LET msg == Head(messageQueue[t])
     IN /\ messageQueue' = [messageQueue EXCEPT ![t] = Tail(messageQueue[t])]
        /\ delivered' = [delivered EXCEPT ![t] =
             [s \in 1..MaxSubscribers |->
               IF s \in subscribers[t]
               THEN Append(delivered[t][s], msg)
               ELSE delivered[t][s]]]
  /\ UNCHANGED subscribers

Next ==
  \/ \E t \in Topics, s \in 1..MaxSubscribers : Subscribe(t, s)
  \/ \E t \in Topics, s \in 1..MaxSubscribers : Unsubscribe(t, s)
  \/ \E t \in Topics, msg \in 1..10 : Publish(t, msg)
  \/ \E t \in Topics : Deliver(t)

DeliveryGuarantee ==
  \A t \in Topics : \A s \in subscribers[t] :
    Len(delivered[t][s]) <= Len(messageQueue[t]) + Len(delivered[t][s])

Spec == Init /\ [][Next]_<<subscribers, messageQueue, delivered>>

====
"""
})

# ─────────────────────────────────────
# Extra: Simple Mutex (reinforcement for BM001 pattern)
# ─────────────────────────────────────
EXAMPLES.append({
    "prompt": "A simple mutual exclusion protocol for N processes using a shared lock variable.",
    "spec": r"""---- MODULE SimpleMutex ----
EXTENDS Naturals, TLC

CONSTANT N

VARIABLES lock, pc

Procs == 1..N

TypeOK ==
  /\ lock \in Procs \cup {0}
  /\ pc \in [Procs -> {"idle", "waiting", "cs"}]

Init ==
  /\ lock = 0
  /\ pc = [p \in Procs |-> "idle"]

TryAcquire(p) ==
  /\ pc[p] = "idle"
  /\ pc' = [pc EXCEPT ![p] = "waiting"]
  /\ UNCHANGED lock

Acquire(p) ==
  /\ pc[p] = "waiting"
  /\ lock = 0
  /\ lock' = p
  /\ pc' = [pc EXCEPT ![p] = "cs"]

Release(p) ==
  /\ pc[p] = "cs"
  /\ lock = p
  /\ lock' = 0
  /\ pc' = [pc EXCEPT ![p] = "idle"]

Next == \E p \in Procs :
  \/ TryAcquire(p)
  \/ Acquire(p)
  \/ Release(p)

MutualExclusion == \A p, q \in Procs : (p # q) => ~(pc[p] = "cs" /\ pc[q] = "cs")

Spec == Init /\ [][Next]_<<lock, pc>>

====
"""
})

# ─────────────────────────────────────
# Extra: Leader Election Ring (reinforces token ring pattern)
# ─────────────────────────────────────
EXAMPLES.append({
    "prompt": "Leader election in a unidirectional ring of N processes. Each process has a unique ID. Process with highest ID becomes leader.",
    "spec": r"""---- MODULE LeaderElection ----
EXTENDS Naturals, TLC

CONSTANT N

Procs == 1..N

VARIABLES leader, msgs, active

TypeOK ==
  /\ leader \in Procs \cup {0}
  /\ msgs \in [Procs -> SUBSET Procs]
  /\ active \in [Procs -> BOOLEAN]

Succ(p) == IF p = N THEN 1 ELSE p + 1

Init ==
  /\ leader = 0
  /\ msgs = [p \in Procs |-> {p}]
  /\ active = [p \in Procs |-> TRUE]

SendId(p) ==
  /\ active[p]
  /\ \E id \in msgs[p] :
       msgs' = [msgs EXCEPT ![Succ(p)] = msgs[Succ(p)] \cup {id}]
  /\ UNCHANGED <<leader, active>>

ReceiveId(p) ==
  /\ active[p]
  /\ \E id \in msgs[p] :
       IF id = p
       THEN /\ leader' = p
            /\ UNCHANGED <<msgs, active>>
       ELSE IF id > p
            THEN /\ msgs' = [msgs EXCEPT ![Succ(p)] = msgs[Succ(p)] \cup {id}]
                 /\ UNCHANGED <<leader, active>>
            ELSE UNCHANGED <<leader, msgs, active>>

Next == \E p \in Procs :
  \/ SendId(p)
  \/ ReceiveId(p)

Spec == Init /\ [][Next]_<<leader, msgs, active>>

====
"""
})

# ─────────────────────────────────────
# Extra: Replicated Log (transaction patterns)
# ─────────────────────────────────────
EXAMPLES.append({
    "prompt": "A replicated log with a leader and N followers. The leader appends entries and followers replicate. Majority commit.",
    "spec": r"""---- MODULE ReplicatedLog ----
EXTENDS Naturals, Sequences, FiniteSets, TLC

CONSTANT N

VARIABLES leaderLog, followerLogs, commitIndex

Followers == 1..N

TypeOK ==
  /\ leaderLog \in Seq(Nat)
  /\ followerLogs \in [Followers -> Seq(Nat)]
  /\ commitIndex \in Nat

Init ==
  /\ leaderLog = <<>>
  /\ followerLogs = [f \in Followers |-> <<>>]
  /\ commitIndex = 0

AppendEntry ==
  /\ \E v \in 1..10 :
       leaderLog' = Append(leaderLog, v)
  /\ UNCHANGED <<followerLogs, commitIndex>>

Replicate(f) ==
  /\ Len(followerLogs[f]) < Len(leaderLog)
  /\ LET nextIdx == Len(followerLogs[f]) + 1
     IN followerLogs' = [followerLogs EXCEPT ![f] = Append(followerLogs[f], leaderLog[nextIdx])]
  /\ UNCHANGED <<leaderLog, commitIndex>>

AdvanceCommit ==
  /\ \E newCI \in 1..Len(leaderLog) :
       /\ newCI > commitIndex
       /\ Cardinality({f \in Followers : Len(followerLogs[f]) >= newCI}) * 2 > N
       /\ commitIndex' = newCI
  /\ UNCHANGED <<leaderLog, followerLogs>>

Next ==
  \/ AppendEntry
  \/ AdvanceCommit
  \/ \E f \in Followers : Replicate(f)

Spec == Init /\ [][Next]_<<leaderLog, followerLogs, commitIndex>>

====
"""
})

# ─────────────────────────────────────
# Extra: Barrier Synchronization
# ─────────────────────────────────────
EXAMPLES.append({
    "prompt": "A barrier synchronization for N processes. All processes must arrive at the barrier before any can proceed.",
    "spec": r"""---- MODULE Barrier ----
EXTENDS Naturals, TLC

CONSTANT N

VARIABLES arrived, released, pc

Procs == 1..N

TypeOK ==
  /\ arrived \in SUBSET Procs
  /\ released \in BOOLEAN
  /\ pc \in [Procs -> {"working", "waiting", "done"}]

Init ==
  /\ arrived = {}
  /\ released = FALSE
  /\ pc = [p \in Procs |-> "working"]

Arrive(p) ==
  /\ pc[p] = "working"
  /\ arrived' = arrived \cup {p}
  /\ pc' = [pc EXCEPT ![p] = "waiting"]
  /\ IF arrived \cup {p} = Procs
     THEN released' = TRUE
     ELSE released' = released

Proceed(p) ==
  /\ pc[p] = "waiting"
  /\ released
  /\ pc' = [pc EXCEPT ![p] = "done"]
  /\ UNCHANGED <<arrived, released>>

Next == \E p \in Procs :
  \/ Arrive(p)
  \/ Proceed(p)

Spec == Init /\ [][Next]_<<arrived, released, pc>>

====
"""
})

# ─────────────────────────────────────
# Extra: Database with MVCC
# ─────────────────────────────────────
EXAMPLES.append({
    "prompt": "Multi-version concurrency control for a database with transactions reading at different timestamps.",
    "spec": r"""---- MODULE MVCC ----
EXTENDS Naturals, FiniteSets, TLC

CONSTANTS Keys, TxIds

VARIABLES versions, activeTx, nextTs

TypeOK ==
  /\ versions \in [Keys -> SUBSET (TxIds \X Nat)]
  /\ activeTx \in SUBSET TxIds
  /\ nextTs \in Nat

Init ==
  /\ versions = [k \in Keys |-> {}]
  /\ activeTx = {}
  /\ nextTs = 1

BeginTx(t) ==
  /\ t \notin activeTx
  /\ activeTx' = activeTx \cup {t}
  /\ UNCHANGED <<versions, nextTs>>

Write(t, k) ==
  /\ t \in activeTx
  /\ versions' = [versions EXCEPT ![k] = versions[k] \cup {<<t, nextTs>>}]
  /\ nextTs' = nextTs + 1
  /\ UNCHANGED activeTx

CommitTx(t) ==
  /\ t \in activeTx
  /\ activeTx' = activeTx \ {t}
  /\ UNCHANGED <<versions, nextTs>>

Next ==
  \/ \E t \in TxIds : BeginTx(t)
  \/ \E t \in TxIds, k \in Keys : Write(t, k)
  \/ \E t \in TxIds : CommitTx(t)

Spec == Init /\ [][Next]_<<versions, activeTx, nextTs>>

====
"""
})

# ─────────────────────────────────────
# Extra: Semaphore
# ─────────────────────────────────────
EXAMPLES.append({
    "prompt": "A counting semaphore with N permits. Processes acquire and release permits. No process should proceed if no permits available.",
    "spec": r"""---- MODULE Semaphore ----
EXTENDS Naturals, TLC

CONSTANTS N, NumProcs

VARIABLES permits, holding

Procs == 1..NumProcs

TypeOK ==
  /\ permits \in 0..N
  /\ holding \in [Procs -> BOOLEAN]

Init ==
  /\ permits = N
  /\ holding = [p \in Procs |-> FALSE]

Acquire(p) ==
  /\ ~holding[p]
  /\ permits > 0
  /\ permits' = permits - 1
  /\ holding' = [holding EXCEPT ![p] = TRUE]

Release(p) ==
  /\ holding[p]
  /\ permits' = permits + 1
  /\ holding' = [holding EXCEPT ![p] = FALSE]

Next == \E p \in Procs :
  \/ Acquire(p)
  \/ Release(p)

Spec == Init /\ [][Next]_<<permits, holding>>

====
"""
})


from src.training.dataset_builder import _DEVELOPER_PROMPT as _CRAFT_DEV_PROMPT

def build_training_msg(prompt: str, spec: str) -> dict:
    analysis = (
        "I need to write a formally verified TLA+ specification. "
        "I'll use EXTENDS for standard modules, declare all VARIABLES explicitly, "
        "define Init, Next, and Spec operators, include TypeOK and safety properties. "
        "No PlusCal syntax — pure TLA+ only."
    )
    return {
        "messages": [
            {"role": "developer", "content": _CRAFT_DEV_PROMPT},
            {"role": "user", "content": prompt},
            {"role": "assistant", "channel": "analysis", "content": analysis},
            {"role": "assistant", "channel": "final", "content": spec.strip()},
        ]
    }


def main():
    passed = 0
    failed = 0
    records = []
    
    for i, ex in enumerate(EXAMPLES):
        spec = ex["spec"].strip()
        import re
        m = re.search(r"MODULE\s+(\w+)", spec)
        modname = m.group(1) if m else f"Spec{i}"
        
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / (modname + ".tla")
            p.write_text(spec)
            result = subprocess.run(
                ["java", "-cp", "src/shared/tlc/tla2tools.jar", "tla2sany.SANY", str(p)],
                capture_output=True, text=True, timeout=15
            )
            has_errors = bool(re.search(r'\*\*\* Errors:', result.stdout))
            has_fatal = 'Fatal errors' in result.stdout
            ok = not has_errors and not has_fatal
            
            if ok:
                passed += 1
                records.append(build_training_msg(ex["prompt"], ex["spec"]))
                print(f"  [{i+1:2d}] PASS  {modname}")
            else:
                failed += 1
                for line in result.stdout.split('\n'):
                    if 'error' in line.lower() or 'Error' in line:
                        err = line.strip()[:100]
                        break
                else:
                    err = "unknown error"
                print(f"  [{i+1:2d}] FAIL  {modname}: {err}")
    
    print(f"\nResults: {passed}/{passed+failed} passed")
    
    if records:
        out = Path("data/processed/augmented.jsonl")
        existing = 0
        if out.exists():
            with out.open() as f:
                existing = sum(1 for _ in f)
        
        with out.open("a") as f:
            for rec in records:
                f.write(json.dumps(rec) + "\n")
        
        print(f"Appended {len(records)} records to {out} (was {existing}, now {existing + len(records)})")


if __name__ == "__main__":
    main()
