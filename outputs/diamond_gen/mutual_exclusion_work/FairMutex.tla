---- MODULE FairMutex ----
(***************************************************************************)
(* Mutex with explicit fairness: a `turn` pointer rotates among waiters    *)
(* in round-robin order, guaranteeing bounded bypass.  Each process enters *)
(* the CS only when the turn pointer points at it.                         *)
(***************************************************************************)
EXTENDS Naturals

N == 3
Procs == 1..N

VARIABLES pc, turn, lock

vars == << pc, turn, lock >>

NoHolder == 0

Init == /\ pc   = [i \in Procs |-> "ncs"]
        /\ turn = 1
        /\ lock = NoHolder

\* Begin requesting.
Request(i) ==
    /\ pc[i] = "ncs"
    /\ pc' = [pc EXCEPT ![i] = "wait"]
    /\ UNCHANGED << turn, lock >>

\* Enter only when turn points at us AND the lock is free.
EnterCS(i) ==
    /\ pc[i] = "wait"
    /\ turn = i
    /\ lock = NoHolder
    /\ lock' = i
    /\ pc' = [pc EXCEPT ![i] = "cs"]
    /\ UNCHANGED turn

\* Release: drop the lock and rotate turn to the next process.
Release(i) ==
    /\ pc[i] = "cs"
    /\ lock = i
    /\ lock' = NoHolder
    /\ turn' = (turn % N) + 1
    /\ pc' = [pc EXCEPT ![i] = "ncs"]

\* If our turn passes us by while we are still in NCS, advance turn so the
\* lock isn't blocked waiting for an idle process.
Skip(i) ==
    /\ turn = i
    /\ pc[i] = "ncs"
    /\ turn' = (turn % N) + 1
    /\ UNCHANGED << pc, lock >>

Idle == UNCHANGED vars

Next == \/ \E i \in Procs : Request(i) \/ EnterCS(i) \/ Release(i) \/ Skip(i)
        \/ Idle

Spec == Init /\ [][Next]_vars

TypeOK ==
    /\ pc   \in [Procs -> {"ncs","wait","cs"}]
    /\ turn \in Procs
    /\ lock \in Procs \cup {NoHolder}
    /\ \A i, j \in Procs : (i # j /\ pc[i] = "cs") => pc[j] # "cs"
    /\ (lock # NoHolder) => pc[lock] = "cs"
====
