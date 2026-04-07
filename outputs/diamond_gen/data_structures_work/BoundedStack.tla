---- MODULE BoundedStack ----
EXTENDS Naturals, Sequences

CONSTANTS K, Vals

VARIABLES stack

vars == << stack >>

Init == stack = << >>

\* Push: append to the top (end of sequence).
Push(v) == /\ Len(stack) < K
           /\ stack' = Append(stack, v)

\* Pop: remove the top (last) element.
Pop == /\ Len(stack) > 0
       /\ stack' = SubSeq(stack, 1, Len(stack) - 1)

Next == \/ \E v \in Vals : Push(v)
        \/ Pop

Spec == Init /\ [][Next]_vars

\* Strong invariant: size bounded and elements valid.
Bounded == /\ Len(stack) \in 0..K
           /\ \A i \in 1..Len(stack) : stack[i] \in Vals

TypeOK == /\ stack \in Seq(Vals)
          /\ Bounded
====
