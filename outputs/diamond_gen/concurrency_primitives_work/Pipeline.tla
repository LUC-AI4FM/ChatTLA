---- MODULE Pipeline ----
EXTENDS Naturals

CONSTANTS K, MaxIn

\* A three-stage pipeline. q1, q2, q3 are bounded queue lengths (capacity K).
\* in_count  : items that have entered stage 1
\* out_count : items that have left stage 3
VARIABLES q1, q2, q3, in_count, out_count

vars == << q1, q2, q3, in_count, out_count >>

Init == /\ q1 = 0
        /\ q2 = 0
        /\ q3 = 0
        /\ in_count  = 0
        /\ out_count = 0

\* External producer feeds stage 1.
Ingest == /\ q1 < K
          /\ in_count < MaxIn
          /\ q1' = q1 + 1
          /\ in_count' = in_count + 1
          /\ UNCHANGED << q2, q3, out_count >>

\* Stage 1 -> stage 2.
Move12 == /\ q1 > 0
          /\ q2 < K
          /\ q1' = q1 - 1
          /\ q2' = q2 + 1
          /\ UNCHANGED << q3, in_count, out_count >>

\* Stage 2 -> stage 3.
Move23 == /\ q2 > 0
          /\ q3 < K
          /\ q2' = q2 - 1
          /\ q3' = q3 + 1
          /\ UNCHANGED << q1, in_count, out_count >>

\* Stage 3 -> output.
Emit == /\ q3 > 0
        /\ q3' = q3 - 1
        /\ out_count' = out_count + 1
        /\ UNCHANGED << q1, q2, in_count >>

\* Drain to allow continued exploration once we've ingested MaxIn.
Drain == /\ in_count = MaxIn
         /\ q1 = 0 /\ q2 = 0 /\ q3 = 0
         /\ out_count = MaxIn
         /\ in_count' = 0
         /\ out_count' = 0
         /\ UNCHANGED << q1, q2, q3 >>

Next == \/ Ingest
        \/ Move12
        \/ Move23
        \/ Emit
        \/ Drain

Spec == Init /\ [][Next]_vars

\* Conservation: items ingested = items still in queues + items emitted.
Conservation == in_count = q1 + q2 + q3 + out_count

\* Each queue stays within capacity.
QueuesBounded == /\ q1 \in 0..K
                 /\ q2 \in 0..K
                 /\ q3 \in 0..K

TypeOK == /\ in_count  \in 0..MaxIn
          /\ out_count \in 0..MaxIn
          /\ QueuesBounded
          /\ Conservation
====
