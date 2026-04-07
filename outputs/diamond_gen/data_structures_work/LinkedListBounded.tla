---- MODULE LinkedListBounded ----
EXTENDS Naturals, Sequences

CONSTANTS K, Vals

\* A bounded singly-linked list modeled as a sequence; head = position 1.
VARIABLES list

vars == << list >>

Init == list = << >>

InSeq(s, v) == \E i \in 1..Len(s) : s[i] = v

\* Insert at front (prepend).
InsertFront(v) == /\ v \in Vals
                  /\ Len(list) < K
                  /\ list' = << v >> \o list

\* Delete the first occurrence of v from the list.
DeleteByValue(v) == /\ InSeq(list, v)
                    /\ LET idx == CHOOSE i \in 1..Len(list) : list[i] = v
                       IN list' = SubSeq(list, 1, idx - 1) \o SubSeq(list, idx + 1, Len(list))

Next == (\E v \in Vals : InsertFront(v)) \/ (\E v \in Vals : DeleteByValue(v))

Spec == Init /\ [][Next]_vars

\* Strong invariant: bounded length, all entries valid.
Bounded == /\ Len(list) \in 0..K
           /\ \A i \in 1..Len(list) : list[i] \in Vals

TypeOK == /\ list \in Seq(Vals)
          /\ Bounded
====
