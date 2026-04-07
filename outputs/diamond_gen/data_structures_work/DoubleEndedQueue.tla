---- MODULE DoubleEndedQueue ----
EXTENDS Naturals, Sequences

CONSTANTS K, Vals

VARIABLES deque

vars == << deque >>

Init == deque = << >>

PushFront(v) == /\ Len(deque) < K
                /\ deque' = << v >> \o deque

PushBack(v) == /\ Len(deque) < K
               /\ deque' = Append(deque, v)

PopFront == /\ Len(deque) > 0
            /\ deque' = Tail(deque)

PopBack == /\ Len(deque) > 0
           /\ deque' = SubSeq(deque, 1, Len(deque) - 1)

Next == \/ \E v \in Vals : PushFront(v)
        \/ \E v \in Vals : PushBack(v)
        \/ PopFront
        \/ PopBack

Spec == Init /\ [][Next]_vars

\* Strong invariant: bounded length, elements valid.
Bounded == /\ Len(deque) \in 0..K
           /\ \A i \in 1..Len(deque) : deque[i] \in Vals

TypeOK == /\ deque \in Seq(Vals)
          /\ Bounded
====
