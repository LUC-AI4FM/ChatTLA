---- MODULE Multiset ----
EXTENDS Naturals

CONSTANTS K, Universe

VARIABLES counts

vars == << counts >>

Init == counts = [x \in Universe |-> 0]

Add(x) == /\ x \in Universe
          /\ counts[x] < K
          /\ counts' = [counts EXCEPT ![x] = @ + 1]

\* Remove clamps at 0.
Remove(x) == /\ x \in Universe
             /\ counts[x] > 0
             /\ counts' = [counts EXCEPT ![x] = @ - 1]

Next == (\E x \in Universe : Add(x)) \/ (\E x \in Universe : Remove(x))

Spec == Init /\ [][Next]_vars

\* Strong invariant: per-element multiplicities bounded.
Bounded == \A x \in Universe : counts[x] \in 0..K

TypeOK == /\ counts \in [Universe -> 0..K]
          /\ Bounded
====
