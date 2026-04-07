---- MODULE SkipListAbstract ----
EXTENDS Naturals, FiniteSets

CONSTANTS K, Universe

\* Abstract skip-list: a sorted set under insert / delete / contains.
VARIABLES elements

vars == << elements >>

Init == elements = {}

Insert(x) == /\ x \in Universe
             /\ x \notin elements
             /\ Cardinality(elements) < K
             /\ elements' = elements \cup {x}

Delete(x) == /\ x \in elements
             /\ elements' = elements \ {x}

\* Contains is an observation.
Contains(x) == /\ x \in elements
               /\ UNCHANGED vars

Next == (\E x \in Universe : Insert(x))
        \/ (\E x \in Universe : Delete(x))
        \/ (\E x \in Universe : Contains(x))

Spec == Init /\ [][Next]_vars

\* Strong invariant: bounded; subset of Universe.
Bounded == /\ Cardinality(elements) \in 0..K
           /\ elements \subseteq Universe

TypeOK == /\ elements \subseteq Universe
          /\ Bounded
====
