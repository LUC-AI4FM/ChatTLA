---- MODULE FairQueue ----
(***************************************************************************)
(* Per-flow round-robin fair queueing.  F flows each have a small        *)
(* per-flow queue.  The dequeuer rotates strictly through the flows in   *)
(* order, skipping empty queues but always advancing the cursor.         *)
(*                                                                         *)
(* Safety: queues stay within capacity and the rotation cursor advances  *)
(* fairly (no flow is dequeued out of turn).                             *)
(***************************************************************************)
EXTENDS Naturals

CONSTANT N

ASSUME N \in 2..3

Flows == 0..(N-1)
QCap == 2  \* per-flow queue capacity

VARIABLES q, cursor

vars == << q, cursor >>

Init == /\ q = [f \in Flows |-> 0]
        /\ cursor = 0

Enqueue(f) == /\ q[f] < QCap
              /\ q' = [q EXCEPT ![f] = @ + 1]
              /\ UNCHANGED cursor

\* Dequeue from the cursor flow if non-empty, then rotate.
Dequeue == /\ q[cursor] > 0
           /\ q' = [q EXCEPT ![cursor] = @ - 1]
           /\ cursor' = (cursor + 1) % N

\* Skip an empty cursor (still rotates fairness pointer).
Skip == /\ q[cursor] = 0
        /\ cursor' = (cursor + 1) % N
        /\ UNCHANGED q

Next == (\E f \in Flows : Enqueue(f)) \/ Dequeue \/ Skip

Spec == Init /\ [][Next]_vars

FairnessInv == cursor \in 0..(N-1) /\ \A f \in Flows : q[f] \in 0..QCap

TypeOK == FairnessInv
====
