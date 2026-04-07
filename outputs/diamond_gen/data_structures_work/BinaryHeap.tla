---- MODULE BinaryHeap ----
EXTENDS Naturals, FiniteSets

CONSTANTS K, MaxElem

\* Element domain: small contiguous range of naturals.
Elems == 1..MaxElem

VARIABLES heap  \* set of elements currently in the heap

vars == << heap >>

Init == heap = {}

\* Min of a non-empty set of naturals.
Min(S) == CHOOSE x \in S : \A y \in S : x <= y

Insert(e) == /\ e \in Elems
             /\ e \notin heap
             /\ Cardinality(heap) < K
             /\ heap' = heap \cup {e}

\* ExtractMin: remove the current minimum.
ExtractMin == /\ heap # {}
              /\ heap' = heap \ { Min(heap) }

Next == (\E e \in Elems : Insert(e)) \/ ExtractMin

Spec == Init /\ [][Next]_vars

\* Strong invariant: bounded; subset of Elems.
Bounded == /\ Cardinality(heap) \in 0..K
           /\ heap \subseteq Elems

TypeOK == /\ heap \subseteq Elems
          /\ Bounded
====
