---- MODULE PriorityQueue ----
EXTENDS Naturals

CONSTANTS K

\* Priorities: 1 (low) .. 3 (high). Pop returns highest non-empty.
Prios == 1..3

VARIABLES counts

vars == << counts >>

Init == counts = [p \in Prios |-> 0]

\* Push at any priority; bounded per level.
Push(p) == /\ counts[p] < K
           /\ counts' = [counts EXCEPT ![p] = @ + 1]

\* Pop highest non-empty priority.
HighestNonEmpty(c) == CHOOSE p \in Prios : c[p] > 0 /\ \A q \in Prios : q > p => c[q] = 0

Pop == /\ \E p \in Prios : counts[p] > 0
       /\ LET p == HighestNonEmpty(counts) IN
            counts' = [counts EXCEPT ![p] = @ - 1]

Next == (\E p \in Prios : Push(p)) \/ Pop

Spec == Init /\ [][Next]_vars

\* Strong invariant: per-priority counts bounded.
Bounded == \A p \in Prios : counts[p] \in 0..K

TypeOK == /\ counts \in [Prios -> 0..K]
          /\ Bounded
====
