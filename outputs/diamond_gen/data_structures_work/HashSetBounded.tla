---- MODULE HashSetBounded ----
EXTENDS Naturals, FiniteSets

CONSTANTS K, Universe

VARIABLES contents

vars == << contents >>

Init == contents = {}

Insert(x) == /\ x \in Universe
             /\ Cardinality(contents) < K
             /\ contents' = contents \cup {x}

Remove(x) == /\ x \in contents
             /\ contents' = contents \ {x}

Next == (\E x \in Universe : Insert(x)) \/ (\E x \in Universe : Remove(x))

Spec == Init /\ [][Next]_vars

\* Strong invariant: bounded size; subset of universe.
Bounded == /\ Cardinality(contents) \in 0..K
           /\ contents \subseteq Universe

TypeOK == /\ contents \subseteq Universe
          /\ Bounded
====
