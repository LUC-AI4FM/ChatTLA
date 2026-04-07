---- MODULE RingBuffer ----
EXTENDS Naturals

CONSTANTS K

VARIABLES head, tail, size

vars == << head, tail, size >>

Init == /\ head = 0
        /\ tail = 0
        /\ size = 0

\* Producer writes at tail; blocks when buffer full.
Produce == /\ size < K
           /\ tail' = (tail + 1) % K
           /\ size' = size + 1
           /\ UNCHANGED head

\* Consumer reads at head; blocks when buffer empty.
Consume == /\ size > 0
           /\ head' = (head + 1) % K
           /\ size' = size - 1
           /\ UNCHANGED tail

Next == Produce \/ Consume

Spec == Init /\ [][Next]_vars

\* Strong invariant: head/tail in 0..K-1, size matches their relationship.
Bounded == /\ size \in 0..K
           /\ head \in 0..(K-1)
           /\ tail \in 0..(K-1)
           /\ ((tail - head) % K) = (size % K)

TypeOK == /\ head \in 0..(K-1)
          /\ tail \in 0..(K-1)
          /\ size \in 0..K
          /\ Bounded
====
