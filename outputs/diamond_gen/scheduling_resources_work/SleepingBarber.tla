---- MODULE SleepingBarber ----
(***************************************************************************)
(* Sleeping barber problem.  One barber, N waiting chairs.                 *)
(* Customers arrive; if all chairs are taken (and the barber is busy),    *)
(* they leave.  Otherwise they sit and wait.  The barber serves one       *)
(* customer at a time and then either takes the next or sleeps.           *)
(*                                                                         *)
(* Safety: barber serves at most one customer at a time; the number of   *)
(* waiting customers never exceeds the number of chairs.                  *)
(***************************************************************************)
EXTENDS Naturals

CONSTANT N

ASSUME N \in 1..3

VARIABLES barber, waiting

\* barber in {"sleeping","cutting"}; waiting in 0..N
vars == << barber, waiting >>

Init == /\ barber  = "sleeping"
        /\ waiting = 0

\* Customer arrives, barber is sleeping → wake him up directly into "cutting".
WakeBarber == /\ barber = "sleeping"
              /\ waiting = 0
              /\ barber' = "cutting"
              /\ UNCHANGED waiting

\* Customer arrives, barber is busy AND a chair is free → sit in waiting room.
JoinQueue == /\ barber = "cutting"
             /\ waiting < N
             /\ waiting' = waiting + 1
             /\ UNCHANGED barber

\* Customer arrives, barber busy and no chairs → leaves (no state change).
\* (Modeled implicitly by the absence of a transition.)

\* Barber finishes a haircut: if a customer is waiting, take them; else sleep.
FinishWithNext == /\ barber = "cutting"
                  /\ waiting > 0
                  /\ waiting' = waiting - 1
                  /\ UNCHANGED barber

FinishSleep == /\ barber = "cutting"
               /\ waiting = 0
               /\ barber' = "sleeping"
               /\ UNCHANGED waiting

Next == WakeBarber \/ JoinQueue \/ FinishWithNext \/ FinishSleep

Spec == Init /\ [][Next]_vars

\* Strong safety: waiting customers never exceed chair capacity, and the
\* barber is sleeping only when no one is waiting.
BarberInv == waiting \in 0..N /\ ((barber = "sleeping") => (waiting = 0))

TypeOK == /\ barber \in {"sleeping","cutting"} /\ BarberInv
====
