---- MODULE Mutex ----
EXTENDS Naturals, FiniteSets

CONSTANT Procs

\* owner is a SET of at most one process; empty means free.
VARIABLE owner

vars == << owner >>

Init == owner = {}

\* Acquire is unconditional when free.
Lock(p) == /\ owner = {}
           /\ owner' = {p}

\* TryLock may succeed (when free) or fail (no state change).
TryLockSucceed(p) == /\ owner = {}
                     /\ owner' = {p}

TryLockFail(p) == /\ owner # {}
                  /\ UNCHANGED owner

\* Only the current owner may release.
Unlock(p) == /\ owner = {p}
             /\ owner' = {}

Next == \/ \E p \in Procs : Lock(p)
        \/ \E p \in Procs : TryLockSucceed(p)
        \/ \E p \in Procs : TryLockFail(p)
        \/ \E p \in Procs : Unlock(p)

Spec == Init /\ [][Next]_vars

\* Safety: at most one owner at any time.
MutexSafe == Cardinality(owner) <= 1

TypeOK == /\ owner \subseteq Procs
          /\ MutexSafe
====
