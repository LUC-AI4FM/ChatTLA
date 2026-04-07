---- MODULE Trie ----
EXTENDS Naturals, Sequences, FiniteSets

\* K = max number of inserted words.
CONSTANTS K

\* Tiny alphabet and word-length cap; words are length-2 strings over {a, b}.
Alphabet == {"a", "b"}

\* All length-2 words.
Words == { << x, y >> : x \in Alphabet, y \in Alphabet }

VARIABLES inserted

vars == << inserted >>

Init == inserted = {}

Insert(w) == /\ w \in Words
             /\ w \notin inserted
             /\ Cardinality(inserted) < K
             /\ inserted' = inserted \cup {w}

\* Lookup: positive iff inserted; state unchanged.
Lookup(w) == /\ w \in inserted
             /\ UNCHANGED vars

Next == (\E w \in Words : Insert(w)) \/ (\E w \in Words : Lookup(w))

Spec == Init /\ [][Next]_vars /\ WF_vars(Next)

\* Strong invariant: bounded; only valid words present.
Bounded == /\ Cardinality(inserted) \in 0..K
           /\ inserted \subseteq Words

TypeOK == /\ inserted \subseteq Words
          /\ Bounded
====
