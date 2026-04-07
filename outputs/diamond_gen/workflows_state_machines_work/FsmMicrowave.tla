---- MODULE FsmMicrowave ----
(***************************************************************************)
(*  Microwave FSM with door interlock.                                     *)
(*    idle -> cooking -> paused -> cooking -> finished                     *)
(*  Cooking is only allowed when the door is closed.                       *)
(***************************************************************************)
EXTENDS Naturals

CONSTANT MaxCookTicks

VARIABLES state, door_closed, cook_time, ever_cooked

vars == << state, door_closed, cook_time, ever_cooked >>

States == {"idle", "cooking", "paused", "finished"}

Init == /\ state       = "idle"
        /\ door_closed = TRUE
        /\ cook_time   = 0
        /\ ever_cooked = FALSE

StartCook == /\ state = "idle"
             /\ door_closed
             /\ state' = "cooking"
             /\ ever_cooked' = TRUE
             /\ UNCHANGED << door_closed, cook_time >>

CookTick == /\ state = "cooking"
            /\ cook_time < MaxCookTicks
            /\ cook_time' = cook_time + 1
            /\ UNCHANGED << state, door_closed, ever_cooked >>

Pause == /\ state = "cooking"
         /\ state' = "paused"
         /\ UNCHANGED << door_closed, cook_time, ever_cooked >>

Resume == /\ state = "paused"
          /\ door_closed
          /\ state' = "cooking"
          /\ UNCHANGED << door_closed, cook_time, ever_cooked >>

OpenDoor == /\ door_closed
            /\ state \in {"idle", "paused"}
            /\ door_closed' = FALSE
            /\ UNCHANGED << state, cook_time, ever_cooked >>

CloseDoor == /\ ~door_closed
             /\ door_closed' = TRUE
             /\ UNCHANGED << state, cook_time, ever_cooked >>

Finish == /\ state = "cooking"
          /\ cook_time = MaxCookTicks
          /\ state' = "finished"
          /\ UNCHANGED << door_closed, cook_time, ever_cooked >>

Done == /\ state = "finished"
        /\ UNCHANGED vars

Next == \/ StartCook \/ CookTick \/ Pause \/ Resume \/ OpenDoor \/ CloseDoor \/ Finish \/ Done

Spec == Init /\ [][Next]_vars

\* Cooking implies door closed; finished implies cook_time > 0 and ever cooked.
SafetyInvariant == ((state = "cooking") => door_closed) /\ ((state = "finished") => (cook_time > 0 /\ ever_cooked)) /\ (ever_cooked => cook_time >= 0)

TypeOK == /\ state \in States
          /\ door_closed \in BOOLEAN
          /\ cook_time \in 0..MaxCookTicks
          /\ ever_cooked \in BOOLEAN
          /\ SafetyInvariant
====
