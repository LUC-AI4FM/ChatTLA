---- MODULE PriorityCeilingMutex ----
(***************************************************************************)
(* Sha-Rajkumar-Lehoczky priority ceiling protocol (1990) for two priority *)
(* levels and one shared mutex.  When a low-priority task holds the lock,  *)
(* it temporarily inherits the ceiling priority (= highest task that may   *)
(* request the resource).  This bounds priority inversion to one critical  *)
(* section length.                                                         *)
(***************************************************************************)
EXTENDS Naturals

Tasks == {"low", "high"}
BasePrio == [low |-> 1, high |-> 2]
Ceiling == 2

VARIABLES pc, holder, effective

vars == << pc, holder, effective >>

NoHolder == "none"

Init == /\ pc        = [t \in Tasks |-> "ncs"]
        /\ holder    = NoHolder
        /\ effective = [t \in Tasks |-> BasePrio[t]]

\* Request the lock.
Request(t) ==
    /\ pc[t] = "ncs"
    /\ pc' = [pc EXCEPT ![t] = "trying"]
    /\ UNCHANGED << holder, effective >>

\* Acquire: lock free.  Holder inherits the ceiling priority.
Acquire(t) ==
    /\ pc[t] = "trying"
    /\ holder = NoHolder
    /\ holder' = t
    /\ effective' = [effective EXCEPT ![t] = Ceiling]
    /\ pc' = [pc EXCEPT ![t] = "cs"]

\* Release: lock back to none, restore base priority.
Release(t) ==
    /\ pc[t] = "cs"
    /\ holder = t
    /\ holder' = NoHolder
    /\ effective' = [effective EXCEPT ![t] = BasePrio[t]]
    /\ pc' = [pc EXCEPT ![t] = "ncs"]

Idle == UNCHANGED vars

Next == \/ \E t \in Tasks : Request(t) \/ Acquire(t) \/ Release(t)
        \/ Idle

Spec == Init /\ [][Next]_vars

\* Mutual exclusion + holder consistency.
TypeOK ==
    /\ pc        \in [Tasks -> {"ncs","trying","cs"}]
    /\ holder    \in Tasks \cup {NoHolder}
    /\ effective \in [Tasks -> 1..Ceiling]
    /\ (holder # NoHolder) => pc[holder] = "cs"
    /\ (holder # NoHolder) => effective[holder] = Ceiling
    /\ \A s, t \in Tasks : (s # t /\ pc[s] = "cs") => pc[t] # "cs"
====
