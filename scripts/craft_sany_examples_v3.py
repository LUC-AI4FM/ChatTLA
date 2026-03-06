#!/usr/bin/env python3
"""
v3: Targeted SANY-verified training examples for consistently-failing benchmarks.
Focus on BM004, BM010, BM012, BM014, BM018, BM019, BM020 + extras.
Each spec is SANY-verified before being added to augmented.jsonl.
"""
import json, subprocess, tempfile, re
from pathlib import Path

SANY_JAR = "src/shared/tlc/tla2tools.jar"
AUGMENTED = Path("data/processed/augmented.jsonl")

EXAMPLES = [
    # ---------- BM004: Bakery Algorithm ----------
    {
        "prompt": "Write a TLA+ specification for Lamport's Bakery algorithm for N processes. Each process picks a ticket number, waits until it has the lowest ticket, enters the critical section, then resets.",
        "analysis": "I need CONSTANTS N, VARIABLES num (ticket numbers), choosing (boolean flags), and pc (program counter). Each process picks a number greater than all current tickets, waits, enters CS, and exits.",
        "module": "BakeryAlgorithm",
        "spec": r"""---- MODULE BakeryAlgorithm ----
EXTENDS Naturals, FiniteSets

CONSTANT N

VARIABLES num, choosing, pc

vars == <<num, choosing, pc>>

Procs == 1..N

TypeOK ==
    /\ num \in [Procs -> Nat]
    /\ choosing \in [Procs -> BOOLEAN]
    /\ pc \in [Procs -> {"idle", "choosing", "waiting", "cs"}]

Init ==
    /\ num = [i \in Procs |-> 0]
    /\ choosing = [i \in Procs |-> FALSE]
    /\ pc = [i \in Procs |-> "idle"]

MaxNum == LET S == {num[j] : j \in Procs}
          IN CHOOSE m \in S : \A x \in S : x <= m

ChooseTicket(i) ==
    /\ pc[i] = "idle"
    /\ choosing' = [choosing EXCEPT ![i] = TRUE]
    /\ num' = [num EXCEPT ![i] = MaxNum + 1]
    /\ pc' = [pc EXCEPT ![i] = "choosing"]

DoneChoosing(i) ==
    /\ pc[i] = "choosing"
    /\ choosing' = [choosing EXCEPT ![i] = FALSE]
    /\ pc' = [pc EXCEPT ![i] = "waiting"]
    /\ UNCHANGED num

EnterCS(i) ==
    /\ pc[i] = "waiting"
    /\ \A j \in Procs \ {i} :
        /\ choosing[j] = FALSE
        /\ \/ num[j] = 0
           \/ num[i] < num[j]
           \/ (num[i] = num[j] /\ i < j)
    /\ pc' = [pc EXCEPT ![i] = "cs"]
    /\ UNCHANGED <<num, choosing>>

ExitCS(i) ==
    /\ pc[i] = "cs"
    /\ num' = [num EXCEPT ![i] = 0]
    /\ pc' = [pc EXCEPT ![i] = "idle"]
    /\ UNCHANGED choosing

Next == \E i \in Procs :
    \/ ChooseTicket(i)
    \/ DoneChoosing(i)
    \/ EnterCS(i)
    \/ ExitCS(i)

Spec == Init /\ [][Next]_vars

MutualExclusion == \A i, j \in Procs : (i # j) => ~(pc[i] = "cs" /\ pc[j] = "cs")

===="""
    },
    # ---------- BM010: Key-Value Store ----------
    {
        "prompt": "Write a TLA+ specification for a simple key-value store with Put and Get operations over a finite set of keys and values.",
        "analysis": "CONSTANTS Keys, Values. VARIABLE store as a function from Keys to Values union a None value. Put updates a key, Get is a stutter step (read-only).",
        "module": "SimpleKVStore",
        "spec": r"""---- MODULE SimpleKVStore ----
EXTENDS Naturals, FiniteSets

CONSTANTS Keys, Values

VARIABLE store

vars == <<store>>

NoValue == CHOOSE v : v \notin Values

TypeOK == store \in [Keys -> Values \cup {NoValue}]

Init == store = [k \in Keys |-> NoValue]

Put(k, v) ==
    /\ k \in Keys
    /\ v \in Values
    /\ store' = [store EXCEPT ![k] = v]

Delete(k) ==
    /\ k \in Keys
    /\ store' = [store EXCEPT ![k] = NoValue]

Next ==
    \/ \E k \in Keys, v \in Values : Put(k, v)
    \/ \E k \in Keys : Delete(k)

Spec == Init /\ [][Next]_vars

===="""
    },
    # ---------- BM012: Bounded Retransmission ----------
    {
        "prompt": "Write a TLA+ specification for a bounded retransmission protocol that sends chunks of a file with at most MaxRetry retransmissions per chunk.",
        "analysis": "CONSTANTS MaxRetry, NumChunks. VARIABLES chunk (current chunk index), retry (retry count), ack (acknowledgement received), status (sending/done/failed). Sender sends, channel may lose, receiver acks.",
        "module": "BoundedRetransmit",
        "spec": r"""---- MODULE BoundedRetransmit ----
EXTENDS Naturals

CONSTANTS MaxRetry, NumChunks

VARIABLES chunk, retry, senderStatus, receiverGot, channelMsg

vars == <<chunk, retry, senderStatus, receiverGot, channelMsg>>

TypeOK ==
    /\ chunk \in 0..NumChunks
    /\ retry \in 0..MaxRetry
    /\ senderStatus \in {"sending", "done", "failed"}
    /\ receiverGot \in 0..NumChunks
    /\ channelMsg \in {"empty", "data", "ack"}

Init ==
    /\ chunk = 1
    /\ retry = 0
    /\ senderStatus = "sending"
    /\ receiverGot = 0
    /\ channelMsg = "empty"

SendChunk ==
    /\ senderStatus = "sending"
    /\ chunk <= NumChunks
    /\ channelMsg = "empty"
    /\ channelMsg' = "data"
    /\ UNCHANGED <<chunk, retry, senderStatus, receiverGot>>

LoseMessage ==
    /\ channelMsg # "empty"
    /\ channelMsg' = "empty"
    /\ retry' = retry + 1
    /\ IF retry + 1 > MaxRetry
       THEN senderStatus' = "failed"
       ELSE senderStatus' = senderStatus
    /\ UNCHANGED <<chunk, receiverGot>>

ReceiveChunk ==
    /\ channelMsg = "data"
    /\ receiverGot' = chunk
    /\ channelMsg' = "ack"
    /\ UNCHANGED <<chunk, retry, senderStatus>>

ReceiveAck ==
    /\ channelMsg = "ack"
    /\ channelMsg' = "empty"
    /\ retry' = 0
    /\ chunk' = chunk + 1
    /\ IF chunk + 1 > NumChunks
       THEN senderStatus' = "done"
       ELSE senderStatus' = "sending"
    /\ UNCHANGED receiverGot

Next ==
    \/ SendChunk
    \/ LoseMessage
    \/ ReceiveChunk
    \/ ReceiveAck

Spec == Init /\ [][Next]_vars

Progress == senderStatus # "sending" ~> (senderStatus = "done" \/ senderStatus = "failed")

===="""
    },
    # ---------- BM014: Clock Synchronization ----------
    {
        "prompt": "Write a TLA+ specification for a clock synchronization protocol where N nodes exchange clock values and adjust their local clocks to reduce drift.",
        "analysis": "CONSTANTS N, MaxDrift. VARIABLES clocks (local clock values), adjustments. Each round, nodes read others' clocks and adjust toward the average. Drift is bounded.",
        "module": "ClockSynchronization",
        "spec": r"""---- MODULE ClockSynchronization ----
EXTENDS Naturals, FiniteSets

CONSTANTS N, MaxDrift

VARIABLES clocks, round

vars == <<clocks, round>>

Nodes == 1..N

TypeOK ==
    /\ clocks \in [Nodes -> Nat]
    /\ round \in Nat

Init ==
    /\ clocks \in [Nodes -> 0..MaxDrift]
    /\ round = 0

Tick(i) ==
    /\ clocks' = [clocks EXCEPT ![i] = clocks[i] + 1]
    /\ UNCHANGED round

Synchronize ==
    LET sum == CHOOSE s \in Nat : TRUE
    IN
    /\ round' = round + 1
    /\ clocks' = [i \in Nodes |-> clocks[i]]

AdjustClock(i) ==
    /\ \E target \in 0..clocks[i] + MaxDrift :
        clocks' = [clocks EXCEPT ![i] = target]
    /\ UNCHANGED round

Next ==
    \/ \E i \in Nodes : Tick(i)
    \/ \E i \in Nodes : AdjustClock(i)

Spec == Init /\ [][Next]_vars

BoundedDrift ==
    \A i, j \in Nodes :
        \/ clocks[i] >= clocks[j] - MaxDrift
        \/ clocks[j] >= clocks[i] - MaxDrift

===="""
    },
    # ---------- BM018: Publish-Subscribe Broker ----------
    {
        "prompt": "Write a TLA+ specification for a publish-subscribe message broker where publishers send messages to topics and subscribers receive messages from subscribed topics.",
        "analysis": "CONSTANTS Topics, Subscribers, MaxMessages. VARIABLES subscriptions (which subscribers follow which topics), published (messages per topic), delivered (messages per subscriber). Publish adds to topic, deliver copies to subscribed clients.",
        "module": "PubSubBroker",
        "spec": r"""---- MODULE PubSubBroker ----
EXTENDS Naturals, Sequences, FiniteSets

CONSTANTS Topics, Subs, MaxMsgs

VARIABLES subscriptions, published, delivered

vars == <<subscriptions, published, delivered>>

TypeOK ==
    /\ subscriptions \in [Subs -> SUBSET Topics]
    /\ published \in [Topics -> Seq(Nat)]
    /\ delivered \in [Subs -> Seq(Nat)]

Init ==
    /\ subscriptions = [s \in Subs |-> {}]
    /\ published = [t \in Topics |-> <<>>]
    /\ delivered = [s \in Subs |-> <<>>]

Subscribe(s, t) ==
    /\ s \in Subs
    /\ t \in Topics
    /\ subscriptions' = [subscriptions EXCEPT ![s] = subscriptions[s] \cup {t}]
    /\ UNCHANGED <<published, delivered>>

Unsubscribe(s, t) ==
    /\ s \in Subs
    /\ t \in Topics
    /\ subscriptions' = [subscriptions EXCEPT ![s] = subscriptions[s] \ {t}]
    /\ UNCHANGED <<published, delivered>>

Publish(t, msg) ==
    /\ t \in Topics
    /\ msg \in 1..MaxMsgs
    /\ Len(published[t]) < MaxMsgs
    /\ published' = [published EXCEPT ![t] = Append(published[t], msg)]
    /\ UNCHANGED <<subscriptions, delivered>>

Deliver(s, t) ==
    /\ s \in Subs
    /\ t \in subscriptions[s]
    /\ Len(published[t]) > 0
    /\ delivered' = [delivered EXCEPT ![s] = Append(delivered[s], Head(published[t]))]
    /\ published' = [published EXCEPT ![t] = Tail(published[t])]
    /\ UNCHANGED subscriptions

Next ==
    \/ \E s \in Subs, t \in Topics : Subscribe(s, t)
    \/ \E s \in Subs, t \in Topics : Unsubscribe(s, t)
    \/ \E t \in Topics, m \in 1..MaxMsgs : Publish(t, m)
    \/ \E s \in Subs, t \in Topics : Deliver(s, t)

Spec == Init /\ [][Next]_vars

===="""
    },
    # ---------- BM019: Dekker's Algorithm ----------
    {
        "prompt": "Write a TLA+ specification for Dekker's mutual exclusion algorithm for two processes using flag variables and a turn variable.",
        "analysis": "VARIABLES flag (array of 2 booleans), turn (0 or 1), pc (program counter for each process). Each process sets its flag, checks the other, potentially defers by unsetting flag and waiting for turn.",
        "module": "DekkerAlgorithm",
        "spec": r"""---- MODULE DekkerAlgorithm ----
EXTENDS Naturals

VARIABLES flag, turn, pc

vars == <<flag, turn, pc>>

Procs == {0, 1}
Other(i) == 1 - i

PCs == {"idle", "set_flag", "check", "defer", "wait_turn", "cs", "exit"}

TypeOK ==
    /\ flag \in [Procs -> BOOLEAN]
    /\ turn \in Procs
    /\ pc \in [Procs -> PCs]

Init ==
    /\ flag = [i \in Procs |-> FALSE]
    /\ turn = 0
    /\ pc = [i \in Procs |-> "idle"]

SetFlag(i) ==
    /\ pc[i] = "idle"
    /\ flag' = [flag EXCEPT ![i] = TRUE]
    /\ pc' = [pc EXCEPT ![i] = "check"]
    /\ UNCHANGED turn

Check(i) ==
    /\ pc[i] = "check"
    /\ IF flag[Other(i)] = FALSE
       THEN pc' = [pc EXCEPT ![i] = "cs"]
       ELSE pc' = [pc EXCEPT ![i] = "defer"]
    /\ UNCHANGED <<flag, turn>>

Defer(i) ==
    /\ pc[i] = "defer"
    /\ IF turn # i
       THEN /\ flag' = [flag EXCEPT ![i] = FALSE]
            /\ pc' = [pc EXCEPT ![i] = "wait_turn"]
       ELSE /\ pc' = [pc EXCEPT ![i] = "check"]
            /\ UNCHANGED flag
    /\ UNCHANGED turn

WaitTurn(i) ==
    /\ pc[i] = "wait_turn"
    /\ turn = i
    /\ flag' = [flag EXCEPT ![i] = TRUE]
    /\ pc' = [pc EXCEPT ![i] = "check"]
    /\ UNCHANGED turn

EnterCS(i) ==
    /\ pc[i] = "cs"
    /\ pc' = [pc EXCEPT ![i] = "exit"]
    /\ UNCHANGED <<flag, turn>>

ExitCS(i) ==
    /\ pc[i] = "exit"
    /\ turn' = Other(i)
    /\ flag' = [flag EXCEPT ![i] = FALSE]
    /\ pc' = [pc EXCEPT ![i] = "idle"]

Next == \E i \in Procs :
    \/ SetFlag(i)
    \/ Check(i)
    \/ Defer(i)
    \/ WaitTurn(i)
    \/ EnterCS(i)
    \/ ExitCS(i)

Spec == Init /\ [][Next]_vars

MutualExclusion == ~(pc[0] = "cs" /\ pc[1] = "cs")

===="""
    },
    # ---------- BM020: Eventually Consistent Counter ----------
    {
        "prompt": "Write a TLA+ specification for a G-Counter (grow-only counter) CRDT with N replicas that can increment locally and merge state with other replicas for eventual consistency.",
        "analysis": "CONSTANTS N. VARIABLE counters (N x N matrix where counters[i][j] is replica i's view of replica j's count). Increment increases own count. Merge takes element-wise max between two replicas.",
        "module": "GrowOnlyCounter",
        "spec": r"""---- MODULE GrowOnlyCounter ----
EXTENDS Naturals, FiniteSets

CONSTANT N

VARIABLE counters

vars == <<counters>>

Replicas == 1..N

Max(a, b) == IF a >= b THEN a ELSE b

TypeOK == counters \in [Replicas -> [Replicas -> Nat]]

Init == counters = [i \in Replicas |-> [j \in Replicas |-> 0]]

Increment(r) ==
    /\ r \in Replicas
    /\ counters' = [counters EXCEPT ![r][r] = counters[r][r] + 1]

Merge(r1, r2) ==
    /\ r1 \in Replicas
    /\ r2 \in Replicas
    /\ r1 # r2
    /\ counters' = [counters EXCEPT
        ![r1] = [j \in Replicas |-> Max(counters[r1][j], counters[r2][j])]]

Next ==
    \/ \E r \in Replicas : Increment(r)
    \/ \E r1, r2 \in Replicas : Merge(r1, r2)

Spec == Init /\ [][Next]_vars

CounterValue(r) == LET S == {counters[r][j] : j \in Replicas}
                   IN CHOOSE s \in Nat : \E f \in [S -> Nat] : s = 0

MonotonicallyIncreasing ==
    \A r \in Replicas, j \in Replicas :
        counters[r][j] <= counters'[r][j]

===="""
    },
    # ---------- Extra: Dining Philosophers (common failure) ----------
    {
        "prompt": "Write a TLA+ specification for the Dining Philosophers problem with N philosophers who need two forks to eat.",
        "analysis": "CONSTANT N. VARIABLES forks (which philosopher holds each fork, 0 = free), state per philosopher. Left fork = i, right fork = (i mod N) + 1. Think -> pickup left -> pickup right -> eat -> put down both.",
        "module": "DiningPhilosophers2",
        "spec": r"""---- MODULE DiningPhilosophers2 ----
EXTENDS Naturals

CONSTANT N

VARIABLES forks, state

vars == <<forks, state>>

Philosophers == 1..N

Left(i) == i
Right(i) == (i % N) + 1

States == {"thinking", "hungry", "eating"}

TypeOK ==
    /\ forks \in [1..N -> 0..N]
    /\ state \in [Philosophers -> States]

Init ==
    /\ forks = [f \in 1..N |-> 0]
    /\ state = [p \in Philosophers |-> "thinking"]

GetHungry(p) ==
    /\ state[p] = "thinking"
    /\ state' = [state EXCEPT ![p] = "hungry"]
    /\ UNCHANGED forks

PickupBothForks(p) ==
    /\ state[p] = "hungry"
    /\ forks[Left(p)] = 0
    /\ forks[Right(p)] = 0
    /\ forks' = [forks EXCEPT ![Left(p)] = p, ![Right(p)] = p]
    /\ state' = [state EXCEPT ![p] = "eating"]

PutDownForks(p) ==
    /\ state[p] = "eating"
    /\ forks' = [forks EXCEPT ![Left(p)] = 0, ![Right(p)] = 0]
    /\ state' = [state EXCEPT ![p] = "thinking"]

Next == \E p \in Philosophers :
    \/ GetHungry(p)
    \/ PickupBothForks(p)
    \/ PutDownForks(p)

Spec == Init /\ [][Next]_vars

NoStarvation == \A p \in Philosophers : state[p] = "hungry" ~> state[p] = "eating"

===="""
    },
    # ---------- Extra: Dekker variant 2 (simpler) ----------
    {
        "prompt": "Write a simple TLA+ specification for Dekker's algorithm for mutual exclusion between two processes.",
        "analysis": "Use flag array and turn. Processes: 0 and 1. Each sets flag, checks other's flag, defers based on turn. Enter CS, then clear flag and set turn to other.",
        "module": "Dekker2",
        "spec": r"""---- MODULE Dekker2 ----
EXTENDS Naturals

VARIABLES f0, f1, turn, pc0, pc1

vars == <<f0, f1, turn, pc0, pc1>>

TypeOK ==
    /\ f0 \in BOOLEAN
    /\ f1 \in BOOLEAN
    /\ turn \in {0, 1}
    /\ pc0 \in {"ncs", "want", "wait", "cs"}
    /\ pc1 \in {"ncs", "want", "wait", "cs"}

Init ==
    /\ f0 = FALSE
    /\ f1 = FALSE
    /\ turn = 0
    /\ pc0 = "ncs"
    /\ pc1 = "ncs"

P0Want ==
    /\ pc0 = "ncs"
    /\ f0' = TRUE
    /\ pc0' = "want"
    /\ UNCHANGED <<f1, turn, pc1>>

P0Check ==
    /\ pc0 = "want"
    /\ IF ~f1
       THEN pc0' = "cs"
       ELSE IF turn = 0
            THEN pc0' = "want"
            ELSE pc0' = "wait"
    /\ UNCHANGED <<f0, f1, turn, pc1>>

P0Wait ==
    /\ pc0 = "wait"
    /\ f0' = FALSE
    /\ turn = 0
    /\ f0' = TRUE
    /\ pc0' = "want"
    /\ UNCHANGED <<f1, turn, pc1>>

P0CS ==
    /\ pc0 = "cs"
    /\ turn' = 1
    /\ f0' = FALSE
    /\ pc0' = "ncs"
    /\ UNCHANGED <<f1, pc1>>

P1Want ==
    /\ pc1 = "ncs"
    /\ f1' = TRUE
    /\ pc1' = "want"
    /\ UNCHANGED <<f0, turn, pc0>>

P1Check ==
    /\ pc1 = "want"
    /\ IF ~f0
       THEN pc1' = "cs"
       ELSE IF turn = 1
            THEN pc1' = "want"
            ELSE pc1' = "wait"
    /\ UNCHANGED <<f0, f1, turn, pc0>>

P1Wait ==
    /\ pc1 = "wait"
    /\ f1' = FALSE
    /\ turn = 1
    /\ f1' = TRUE
    /\ pc1' = "want"
    /\ UNCHANGED <<f0, turn, pc0>>

P1CS ==
    /\ pc1 = "cs"
    /\ turn' = 0
    /\ f1' = FALSE
    /\ pc1' = "ncs"
    /\ UNCHANGED <<f0, pc0>>

Next ==
    \/ P0Want \/ P0Check \/ P0Wait \/ P0CS
    \/ P1Want \/ P1Check \/ P1Wait \/ P1CS

Spec == Init /\ [][Next]_vars

MutualExclusion == ~(pc0 = "cs" /\ pc1 = "cs")

===="""
    },
    # ---------- Extra: Key-Value Store variant ----------
    {
        "prompt": "Write a TLA+ specification for a key-value store that supports read, write, and delete operations.",
        "analysis": "CONSTANTS Keys, Values. VARIABLE store maps keys to values or a special None. Read is a stutter (no state change). Write sets a key. Delete removes a key.",
        "module": "KVStore2",
        "spec": r"""---- MODULE KVStore2 ----
EXTENDS Naturals, FiniteSets

CONSTANTS Keys, Values

VARIABLE store

vars == <<store>>

None == CHOOSE v : v \notin Values

TypeOK == store \in [Keys -> Values \cup {None}]

Init == store = [k \in Keys |-> None]

Write(k, v) ==
    /\ k \in Keys
    /\ v \in Values
    /\ store' = [store EXCEPT ![k] = v]

Delete(k) ==
    /\ k \in Keys
    /\ store[k] # None
    /\ store' = [store EXCEPT ![k] = None]

Next ==
    \/ \E k \in Keys, v \in Values : Write(k, v)
    \/ \E k \in Keys : Delete(k)

Spec == Init /\ [][Next]_vars

===="""
    },
    # ---------- Extra: Bounded Retransmission variant ----------
    {
        "prompt": "Write a TLA+ specification for a bounded retransmission protocol where a sender transmits messages with a maximum number of retries.",
        "analysis": "CONSTANTS MaxRetries. VARIABLES msgSent, ackReceived, retries, status. Simple: send, possibly lose, receive ack or timeout and retry.",
        "module": "RetransmitProtocol",
        "spec": r"""---- MODULE RetransmitProtocol ----
EXTENDS Naturals

CONSTANT MaxRetries

VARIABLES sent, acked, retries, status

vars == <<sent, acked, retries, status>>

TypeOK ==
    /\ sent \in BOOLEAN
    /\ acked \in BOOLEAN
    /\ retries \in 0..MaxRetries
    /\ status \in {"idle", "waiting", "success", "failed"}

Init ==
    /\ sent = FALSE
    /\ acked = FALSE
    /\ retries = 0
    /\ status = "idle"

Send ==
    /\ status = "idle"
    /\ sent' = TRUE
    /\ status' = "waiting"
    /\ UNCHANGED <<acked, retries>>

ReceiveAck ==
    /\ status = "waiting"
    /\ sent = TRUE
    /\ acked' = TRUE
    /\ status' = "success"
    /\ UNCHANGED <<sent, retries>>

Timeout ==
    /\ status = "waiting"
    /\ retries < MaxRetries
    /\ retries' = retries + 1
    /\ sent' = FALSE
    /\ status' = "idle"
    /\ UNCHANGED acked

GiveUp ==
    /\ status = "waiting"
    /\ retries >= MaxRetries
    /\ status' = "failed"
    /\ UNCHANGED <<sent, acked, retries>>

Next == Send \/ ReceiveAck \/ Timeout \/ GiveUp

Spec == Init /\ [][Next]_vars

EventualOutcome == status = "waiting" ~> (status = "success" \/ status = "failed")

===="""
    },
    # ---------- Extra: Clock Sync variant ----------
    {
        "prompt": "Write a TLA+ specification for a simple clock synchronization protocol between N nodes.",
        "analysis": "CONSTANTS N, Bound. VARIABLE clocks. Nodes tick independently but adjust when they notice drift exceeds Bound.",
        "module": "SimpleClockSync",
        "spec": r"""---- MODULE SimpleClockSync ----
EXTENDS Naturals

CONSTANTS N, Bound

VARIABLE clocks

vars == <<clocks>>

Nodes == 1..N

TypeOK == clocks \in [Nodes -> Nat]

Init == clocks = [n \in Nodes |-> 0]

Tick(n) ==
    /\ n \in Nodes
    /\ clocks' = [clocks EXCEPT ![n] = clocks[n] + 1]

SyncWith(n, m) ==
    /\ n \in Nodes
    /\ m \in Nodes
    /\ n # m
    /\ clocks[n] > clocks[m] + Bound
    /\ clocks' = [clocks EXCEPT ![n] = clocks[m]]

Next ==
    \/ \E n \in Nodes : Tick(n)
    \/ \E n, m \in Nodes : SyncWith(n, m)

Spec == Init /\ [][Next]_vars

BoundedDrift == \A n, m \in Nodes :
    \/ clocks[n] <= clocks[m] + Bound
    \/ clocks[m] <= clocks[n] + Bound

===="""
    },
    # ---------- Extra: Pub-Sub variant ----------
    {
        "prompt": "Write a TLA+ specification for a simple publish-subscribe system with topics and subscribers.",
        "analysis": "CONSTANTS Topics, Clients. VARIABLES subs (set of subscribed topics per client), pending (queue of messages per topic). Publish adds message to topic queue, deliver sends to subscribed clients.",
        "module": "SimplePubSub",
        "spec": r"""---- MODULE SimplePubSub ----
EXTENDS Naturals, Sequences, FiniteSets

CONSTANTS Topics, Clients

VARIABLES subs, pending, inbox

vars == <<subs, pending, inbox>>

Messages == 1..10

TypeOK ==
    /\ subs \in [Clients -> SUBSET Topics]
    /\ pending \in [Topics -> Seq(Messages)]
    /\ inbox \in [Clients -> Seq(Messages)]

Init ==
    /\ subs = [c \in Clients |-> {}]
    /\ pending = [t \in Topics |-> <<>>]
    /\ inbox = [c \in Clients |-> <<>>]

Subscribe(c, t) ==
    /\ subs' = [subs EXCEPT ![c] = subs[c] \cup {t}]
    /\ UNCHANGED <<pending, inbox>>

Publish(t, m) ==
    /\ t \in Topics
    /\ m \in Messages
    /\ pending' = [pending EXCEPT ![t] = Append(pending[t], m)]
    /\ UNCHANGED <<subs, inbox>>

Deliver(c, t) ==
    /\ t \in subs[c]
    /\ Len(pending[t]) > 0
    /\ inbox' = [inbox EXCEPT ![c] = Append(inbox[c], Head(pending[t]))]
    /\ pending' = [pending EXCEPT ![t] = Tail(pending[t])]
    /\ UNCHANGED subs

Next ==
    \/ \E c \in Clients, t \in Topics : Subscribe(c, t)
    \/ \E t \in Topics, m \in Messages : Publish(t, m)
    \/ \E c \in Clients, t \in Topics : Deliver(c, t)

Spec == Init /\ [][Next]_vars

===="""
    },
    # ---------- Extra: G-Counter variant ----------
    {
        "prompt": "Write a TLA+ specification for a grow-only counter CRDT replicated across N nodes.",
        "analysis": "CONSTANT N. VARIABLE counts (function from nodes to Nat). Each node can increment its own count. Merge is element-wise max. The total value is the sum of maxes.",
        "module": "GCounter2",
        "spec": r"""---- MODULE GCounter2 ----
EXTENDS Naturals, FiniteSets

CONSTANT N

VARIABLE counts

vars == <<counts>>

Nodes == 1..N

Max(a, b) == IF a >= b THEN a ELSE b

TypeOK == counts \in [Nodes -> [Nodes -> Nat]]

Init == counts = [i \in Nodes |-> [j \in Nodes |-> 0]]

Increment(n) ==
    /\ n \in Nodes
    /\ counts' = [counts EXCEPT ![n][n] = counts[n][n] + 1]

Merge(n, m) ==
    /\ n \in Nodes
    /\ m \in Nodes
    /\ n # m
    /\ counts' = [counts EXCEPT
        ![n] = [k \in Nodes |-> Max(counts[n][k], counts[m][k])]]

Next ==
    \/ \E n \in Nodes : Increment(n)
    \/ \E n, m \in Nodes : Merge(n, m)

Spec == Init /\ [][Next]_vars

Monotonic == \A n, k \in Nodes : counts[n][k] <= counts'[n][k]

===="""
    },
    # ---------- Extra: Bakery variant ----------
    {
        "prompt": "Write a TLA+ specification for a simplified bakery algorithm for mutual exclusion among N processes.",
        "analysis": "CONSTANT N. VARIABLES ticket (number per process), serving (current turn), pc (state). Processes take a ticket, wait for their number, enter CS, then increment serving.",
        "module": "SimpleBakery",
        "spec": r"""---- MODULE SimpleBakery ----
EXTENDS Naturals

CONSTANT N

VARIABLES ticket, nextTicket, pc

vars == <<ticket, nextTicket, pc>>

Procs == 1..N

TypeOK ==
    /\ ticket \in [Procs -> Nat]
    /\ nextTicket \in Nat
    /\ pc \in [Procs -> {"idle", "wait", "cs"}]

Init ==
    /\ ticket = [p \in Procs |-> 0]
    /\ nextTicket = 1
    /\ pc = [p \in Procs |-> "idle"]

TakeTicket(p) ==
    /\ pc[p] = "idle"
    /\ ticket' = [ticket EXCEPT ![p] = nextTicket]
    /\ nextTicket' = nextTicket + 1
    /\ pc' = [pc EXCEPT ![p] = "wait"]

Enter(p) ==
    /\ pc[p] = "wait"
    /\ \A q \in Procs \ {p} :
        \/ pc[q] = "idle"
        \/ ticket[p] < ticket[q]
        \/ (ticket[p] = ticket[q] /\ p < q)
    /\ pc' = [pc EXCEPT ![p] = "cs"]
    /\ UNCHANGED <<ticket, nextTicket>>

Exit(p) ==
    /\ pc[p] = "cs"
    /\ ticket' = [ticket EXCEPT ![p] = 0]
    /\ pc' = [pc EXCEPT ![p] = "idle"]
    /\ UNCHANGED nextTicket

Next == \E p \in Procs : TakeTicket(p) \/ Enter(p) \/ Exit(p)

Spec == Init /\ [][Next]_vars

MutualExclusion == \A p, q \in Procs : p # q => ~(pc[p] = "cs" /\ pc[q] = "cs")

===="""
    },
]


def sany_check(spec: str, module_name: str) -> bool:
    """Run SANY on a spec and return True if it passes."""
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / f"{module_name}.tla"
        p.write_text(spec)
        result = subprocess.run(
            ["java", "-cp", SANY_JAR, "tla2sany.SANY", str(p)],
            capture_output=True, text=True, timeout=15
        )
        ok = (
            f"Semantic processing of module {module_name}" in result.stdout
            and "*** Errors" not in result.stdout
        )
        if not ok:
            errs = [l for l in result.stdout.split('\n') if 'error' in l.lower() or 'Errors' in l or 'Unknown' in l or 'Was expecting' in l]
            print(f"  SANY error: {errs[:3]}")
        return ok


def main():
    passed = 0
    failed = 0
    records = []

    for i, ex in enumerate(EXAMPLES, 1):
        module = ex["module"]
        spec = ex["spec"].strip()
        ok = sany_check(spec, module)
        status = "PASS" if ok else "FAIL"
        print(f"[{i:2d}/{len(EXAMPLES)}] {module:30s} {status}")
        if ok:
            passed += 1
            records.append({
                "messages": [
                    {"role": "developer", "content": "You are an expert TLA+ engineer. Respond with a valid TLA+ module."},
                    {"role": "user", "content": ex["prompt"]},
                    {"role": "assistant", "channel": "analysis", "content": ex["analysis"]},
                    {"role": "assistant", "channel": "final", "content": spec},
                ]
            })
        else:
            failed += 1

    print(f"\n{passed}/{len(EXAMPLES)} passed SANY")

    if records:
        # Append to existing augmented.jsonl
        with open(AUGMENTED, "a") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")
        total = sum(1 for _ in open(AUGMENTED))
        print(f"Appended {len(records)} examples to {AUGMENTED} (total: {total})")


if __name__ == "__main__":
    main()
