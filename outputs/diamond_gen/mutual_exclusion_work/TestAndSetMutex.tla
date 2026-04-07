---- MODULE TestAndSetMutex ----
(***************************************************************************)
(* Spinlock built on a single test-and-set bit.  TAS atomically reads the  *)
(* old value and writes TRUE; only the thread that observed FALSE acquires *)
(* the lock.  Release simply writes FALSE.                                 *)
(***************************************************************************)
EXTENDS Naturals

N == 3
Procs == 1..N

VARIABLES pc, lock

vars == << pc, lock >>

Init == /\ pc   = [i \in Procs |-> "ncs"]
        /\ lock = FALSE

\* Begin trying.
Try(i) ==
    /\ pc[i] = "ncs"
    /\ pc' = [pc EXCEPT ![i] = "tas"]
    /\ UNCHANGED lock

\* Atomic test-and-set: succeed iff lock was FALSE.
TASsucceed(i) ==
    /\ pc[i] = "tas"
    /\ lock = FALSE
    /\ lock' = TRUE
    /\ pc' = [pc EXCEPT ![i] = "cs"]

\* TAS observed lock=TRUE: spin and try again.
TASfail(i) ==
    /\ pc[i] = "tas"
    /\ lock = TRUE
    /\ pc' = [pc EXCEPT ![i] = "tas"]
    /\ UNCHANGED lock

Release(i) ==
    /\ pc[i] = "cs"
    /\ lock' = FALSE
    /\ pc'   = [pc EXCEPT ![i] = "ncs"]

Idle == UNCHANGED vars

Next == \/ \E i \in Procs : Try(i) \/ TASsucceed(i) \/ TASfail(i) \/ Release(i)
        \/ Idle

Spec == Init /\ [][Next]_vars

TypeOK ==
    /\ pc   \in [Procs -> {"ncs","tas","cs"}]
    /\ lock \in BOOLEAN
    /\ \A i, j \in Procs : (i # j /\ pc[i] = "cs") => pc[j] # "cs"
====
