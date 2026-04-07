---- MODULE ChainReplication ----
(***************************************************************************)
(*  Chain replication (van Renesse & Schneider).  Three replicas form   *)
(*  a chain head -> middle -> tail.  Writes enter at the head and flow  *)
(*  along the chain; reads are served from the tail.                    *)
(*                                                                         *)
(*  Strong invariant: each downstream replica's state is a prefix of   *)
(*  its upstream neighbour, and the tail's state is a prefix of the    *)
(*  head's.                                                              *)
(***************************************************************************)
EXTENDS Naturals

CONSTANTS MaxLog

VARIABLES head, middle, tail

vars == << head, middle, tail >>

Init == /\ head   = 0
        /\ middle = 0
        /\ tail   = 0

\* Client write enters at the head.
ClientWrite == /\ head < MaxLog
               /\ head' = head + 1
               /\ UNCHANGED << middle, tail >>

\* The head -> middle hop forwards the next entry.
HeadToMiddle == /\ middle < head
                /\ middle' = middle + 1
                /\ UNCHANGED << head, tail >>

\* The middle -> tail hop forwards the next entry.
MiddleToTail == /\ tail < middle
                /\ tail' = tail + 1
                /\ UNCHANGED << head, middle >>

Next == \/ ClientWrite \/ HeadToMiddle \/ MiddleToTail

Spec == Init /\ [][Next]_vars /\ WF_vars(Next)

\* Strong invariant: tail <= middle <= head.  Conjoined into TypeOK.
TypeOK == /\ head   \in 0..MaxLog
          /\ middle \in 0..MaxLog
          /\ tail   \in 0..MaxLog
          /\ tail   <= middle
          /\ middle <= head
====
