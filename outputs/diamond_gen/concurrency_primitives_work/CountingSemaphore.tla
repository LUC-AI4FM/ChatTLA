---- MODULE CountingSemaphore ----
EXTENDS Naturals, FiniteSets

CONSTANTS Procs, K

VARIABLES count, holders

vars == << count, holders >>

Init == /\ count = K
        /\ holders = {}

\* Acquire: decrement count, mark p as a holder. Blocks at 0.
Acquire(p) == /\ count > 0
              /\ p \notin holders
              /\ count' = count - 1
              /\ holders' = holders \cup {p}

\* Release: only a current holder may release; increment count, capped at K.
Release(p) == /\ p \in holders
              /\ count < K
              /\ count' = count + 1
              /\ holders' = holders \ {p}

Next == \/ \E p \in Procs : Acquire(p)
        \/ \E p \in Procs : Release(p)

Spec == Init /\ [][Next]_vars

\* Strong invariant: count + |holders| = K (conservation), and count in 0..K.
Conservation == count + Cardinality(holders) = K

TypeOK == /\ count \in 0..K
          /\ holders \subseteq Procs
          /\ Conservation
====
