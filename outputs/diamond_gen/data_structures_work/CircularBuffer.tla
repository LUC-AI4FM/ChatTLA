---- MODULE CircularBuffer ----
EXTENDS Naturals

CONSTANTS K, MaxOps

\* Multi-producer/multi-consumer circular buffer modeled with monotone counters.
\* head = total items consumed; tail = total items produced; size = tail - head.
VARIABLES head, tail

vars == << head, tail >>

Init == /\ head = 0
        /\ tail = 0

\* Producer: tail++ when not full.
Produce == /\ tail - head < K
           /\ tail < MaxOps
           /\ tail' = tail + 1
           /\ UNCHANGED head

\* Consumer: head++ when not empty.
Consume == /\ head < tail
           /\ head' = head + 1
           /\ UNCHANGED tail

Next == Produce \/ Consume

Spec == Init /\ [][Next]_vars /\ WF_vars(Next)

\* Strong invariant: head <= tail and the gap never exceeds K.
Bounded == /\ head <= tail
           /\ tail - head <= K
           /\ head \in 0..MaxOps
           /\ tail \in 0..MaxOps

TypeOK == /\ head \in Nat
          /\ tail \in Nat
          /\ Bounded
====
