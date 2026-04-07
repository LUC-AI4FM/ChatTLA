---- MODULE BackpressureChannel ----
(***************************************************************************)
(* Bounded channel with credit-based backpressure.  The sender starts    *)
(* with K credits; each send consumes one credit and the receiver        *)
(* returns one credit per consumed message.                              *)
(*                                                                         *)
(* Safety: in-flight messages never exceed initial credits K.            *)
(***************************************************************************)
EXTENDS Naturals

K == 3  \* initial credits / channel capacity

VARIABLES credits, inflight

vars == << credits, inflight >>

Init == /\ credits  = K
        /\ inflight = 0

\* Send: spend one credit, message becomes in-flight.
Send == /\ credits > 0
        /\ credits'  = credits - 1
        /\ inflight' = inflight + 1

\* Receive: consume an in-flight message, return one credit.
Receive == /\ inflight > 0
           /\ inflight' = inflight - 1
           /\ credits'  = credits + 1

Next == Send \/ Receive

Spec == Init /\ [][Next]_vars

\* Strong safety: credit conservation — credits + in-flight is the constant K.
CreditInv == credits + inflight = K /\ inflight \in 0..K

TypeOK == credits \in 0..K /\ CreditInv
====
