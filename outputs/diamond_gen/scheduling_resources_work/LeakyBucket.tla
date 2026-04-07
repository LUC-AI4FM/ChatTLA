---- MODULE LeakyBucket ----
(***************************************************************************)
(* Leaky-bucket traffic shaper.  Bucket has capacity K; it drains one    *)
(* unit per tick.  Arrivals beyond capacity are dropped.                 *)
(*                                                                         *)
(* Safety: queue length stays in 0..K at all times; dropped counter only *)
(* increases when the bucket is full.                                    *)
(***************************************************************************)
EXTENDS Naturals

K == 3  \* bucket capacity

VARIABLES queue, dropped

vars == << queue, dropped >>

Init == /\ queue = 0
        /\ dropped = 0

\* Arrival admitted: bucket has room.
Arrive == /\ queue < K
          /\ queue' = queue + 1
          /\ UNCHANGED dropped

\* Arrival dropped: bucket full, drop counter increments.
Drop == /\ queue = K
        /\ dropped < K   \* keep state space finite
        /\ dropped' = dropped + 1
        /\ UNCHANGED queue

\* Leak: drain one unit per tick.
Leak == /\ queue > 0
        /\ queue' = queue - 1
        /\ UNCHANGED dropped

\* Reset the dropped counter (small wrap so the model stays finite).
ResetDropped == /\ dropped > 0
                /\ dropped' = 0
                /\ UNCHANGED queue

Next == Arrive \/ Drop \/ Leak \/ ResetDropped

Spec == Init /\ [][Next]_vars

BucketInv == queue \in 0..K /\ dropped \in 0..K

TypeOK == BucketInv
====
