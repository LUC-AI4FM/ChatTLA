---- MODULE BoundedQueue ----
EXTENDS Naturals, Sequences

CONSTANTS K, Vals

VARIABLES queue

vars == << queue >>

Init == queue = << >>

\* Enqueue blocks when full.
Enqueue(v) == /\ Len(queue) < K
              /\ queue' = Append(queue, v)

\* Dequeue blocks when empty; FIFO removes the head.
Dequeue == /\ Len(queue) > 0
           /\ queue' = Tail(queue)

Next == \/ \E v \in Vals : Enqueue(v)
        \/ Dequeue

Spec == Init /\ [][Next]_vars

\* Strong invariant: bounded length AND every element is a valid value.
Bounded == /\ Len(queue) \in 0..K
           /\ \A i \in 1..Len(queue) : queue[i] \in Vals

TypeOK == /\ queue \in Seq(Vals)
          /\ Bounded
====
