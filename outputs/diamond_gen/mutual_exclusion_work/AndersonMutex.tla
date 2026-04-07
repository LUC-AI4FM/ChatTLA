---- MODULE AndersonMutex ----
(***************************************************************************)
(* Anderson's array-based queueing lock (1990).  Each process spins on its *)
(* own slot of a circular flag array `slots`.  An atomic fetch-and-add on  *)
(* `tail` assigns each waiter a unique position.  Local spinning makes the *)
(* lock cache-friendly and gives FIFO ordering.                            *)
(***************************************************************************)
EXTENDS Naturals

N == 2
Procs == 1..N
Size == 2  \* slot array size; >= max concurrent waiters.

VARIABLES pc, slots, tail, mySlot

vars == << pc, slots, tail, mySlot >>

Init == /\ pc     = [i \in Procs |-> "ncs"]
        /\ slots  = [s \in 0..(Size-1) |-> IF s = 0 THEN "go" ELSE "wait"]
        /\ tail   = 0
        /\ mySlot = [i \in Procs |-> 0]

\* Atomic fetch-and-add on tail; remember our slot.
Acquire(i) ==
    /\ pc[i] = "ncs"
    /\ tail < Size  \* bound the queue (state space)
    /\ mySlot' = [mySlot EXCEPT ![i] = tail]
    /\ tail'   = tail + 1
    /\ pc'     = [pc EXCEPT ![i] = "spin"]
    /\ UNCHANGED slots

\* Spin on our own slot until it becomes "go".
EnterCS(i) ==
    /\ pc[i] = "spin"
    /\ slots[mySlot[i]] = "go"
    /\ pc' = [pc EXCEPT ![i] = "cs"]
    /\ UNCHANGED << slots, tail, mySlot >>

\* Release: clear our slot, mark next slot as "go".
Release(i) ==
    /\ pc[i] = "cs"
    /\ LET cur  == mySlot[i]
           nxt  == (cur + 1) % Size
       IN  slots' = [slots EXCEPT ![cur] = "wait", ![nxt] = "go"]
    /\ pc' = [pc EXCEPT ![i] = "ncs"]
    /\ UNCHANGED << tail, mySlot >>

Idle == UNCHANGED vars

Next == \/ \E i \in Procs : Acquire(i) \/ EnterCS(i) \/ Release(i)
        \/ Idle

Spec == Init /\ [][Next]_vars

TypeOK ==
    /\ pc     \in [Procs -> {"ncs","spin","cs"}]
    /\ slots  \in [0..(Size-1) -> {"wait","go"}]
    /\ tail   \in 0..Size
    /\ mySlot \in [Procs -> 0..(Size-1)]
    /\ \A i, j \in Procs : (i # j /\ pc[i] = "cs") => pc[j] # "cs"
====
