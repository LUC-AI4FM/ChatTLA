---- MODULE BinarySemaphore ----
EXTENDS Naturals, FiniteSets

CONSTANT Procs

VARIABLES holder, waiters

vars == << holder, waiters >>

\* "none" means semaphore is free; otherwise the holder is a process id.
NoHolder == "none"

Init == /\ holder = NoHolder
        /\ waiters = {}

\* P(): if free, take it; otherwise enqueue self into waiters set.
Acquire(p) == /\ holder = NoHolder
              /\ p \notin waiters
              /\ holder' = p
              /\ UNCHANGED waiters

Wait(p) == /\ holder # NoHolder
           /\ holder # p
           /\ p \notin waiters
           /\ waiters' = waiters \cup {p}
           /\ UNCHANGED holder

\* V(): release the semaphore. Either hand off to a waiter, or set free.
ReleaseToWaiter(p) == /\ holder = p
                     /\ waiters # {}
                     /\ \E q \in waiters :
                          /\ holder' = q
                          /\ waiters' = waiters \ {q}

ReleaseFree(p) == /\ holder = p
                  /\ waiters = {}
                  /\ holder' = NoHolder
                  /\ UNCHANGED waiters

Next == \/ \E p \in Procs : Acquire(p)
        \/ \E p \in Procs : Wait(p)
        \/ \E p \in Procs : ReleaseToWaiter(p)
        \/ \E p \in Procs : ReleaseFree(p)

Spec == Init /\ [][Next]_vars

\* Strong safety: at most one holder, and a waiting process is not the holder.
MutexSafe == /\ (holder = NoHolder) \/ (holder \in Procs)
             /\ holder \notin waiters

TypeOK == /\ holder \in (Procs \cup {NoHolder})
          /\ waiters \subseteq Procs
          /\ MutexSafe
====
