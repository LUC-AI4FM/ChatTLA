---- MODULE BatchAdmission ----
(***************************************************************************)
(* Batch admission control.  In each tick, up to B requests are admitted *)
(* together.  At a tick boundary the per-tick counter resets.            *)
(*                                                                         *)
(* Safety: per-tick admitted count never exceeds B.                      *)
(***************************************************************************)
EXTENDS Naturals

B == 2  \* batch size per tick

VARIABLES this_tick, total

vars == << this_tick, total >>

Init == /\ this_tick = 0
        /\ total     = 0

\* Admit one request within the current tick.
Admit == /\ this_tick < B
         /\ this_tick' = this_tick + 1
         /\ total'     = (total + 1) % (2 * B + 1)

\* Tick boundary: reset the per-tick counter.
TickBoundary == /\ this_tick' = 0
                /\ UNCHANGED total

Next == Admit \/ TickBoundary

Spec == Init /\ [][Next]_vars

\* Strong safety: per-tick admitted count never exceeds B.
BatchInv == this_tick \in 0..B /\ total \in 0..(2 * B)

TypeOK == BatchInv
====
