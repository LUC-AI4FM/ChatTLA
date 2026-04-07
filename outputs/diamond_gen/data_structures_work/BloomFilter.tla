---- MODULE BloomFilter ----
EXTENDS Naturals, FiniteSets

CONSTANTS Universe

\* Bit positions of the underlying bit array.
Bits == 1..3

\* Two hash functions over the universe → bit positions.
\* Modeled abstractly as fixed mappings (deterministic).
H1(x) == 1
H2(x) == 2

VARIABLES bits, inserted

vars == << bits, inserted >>

Init == /\ bits = {}
        /\ inserted = {}

Insert(x) == /\ x \in Universe
             /\ x \notin inserted
             /\ bits' = bits \cup {H1(x), H2(x)}
             /\ inserted' = inserted \cup {x}

\* Query is positive iff both hash bits are set; we don't change state.
Query(x) == /\ x \in Universe
            /\ {H1(x), H2(x)} \subseteq bits
            /\ UNCHANGED vars

Next == (\E x \in Universe : Insert(x)) \/ (\E x \in Universe : Query(x))

Spec == Init /\ [][Next]_vars

\* Strong invariant — NO FALSE NEGATIVES: every inserted item's bits are set.
NoFalseNegatives == \A x \in inserted : {H1(x), H2(x)} \subseteq bits

TypeOK == /\ bits \subseteq Bits
          /\ inserted \subseteq Universe
          /\ NoFalseNegatives
====
