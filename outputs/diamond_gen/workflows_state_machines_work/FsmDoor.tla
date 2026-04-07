---- MODULE FsmDoor ----
(***************************************************************************)
(*  Door FSM: closed -> opening -> open -> closing -> closed               *)
(*  Sensor events drive transitions; previous-state tracking lets us       *)
(*  encode "opening implies prior state was closed" as an invariant.       *)
(***************************************************************************)
EXTENDS Naturals

CONSTANT MaxCycles

VARIABLES state, prev_state, cycles

vars == << state, prev_state, cycles >>

States == {"closed", "opening", "open", "closing"}

Init == /\ state      = "closed"
        /\ prev_state = "closed"
        /\ cycles     = 0

StartOpen == /\ state = "closed"
             /\ cycles < MaxCycles
             /\ state' = "opening"
             /\ prev_state' = state
             /\ UNCHANGED cycles

FullyOpen == /\ state = "opening"
             /\ state' = "open"
             /\ prev_state' = state
             /\ UNCHANGED cycles

StartClose == /\ state = "open"
              /\ state' = "closing"
              /\ prev_state' = state
              /\ UNCHANGED cycles

FullyClosed == /\ state = "closing"
               /\ state' = "closed"
               /\ prev_state' = state
               /\ cycles' = cycles + 1

Done == /\ cycles = MaxCycles
        /\ state = "closed"
        /\ UNCHANGED vars

Next == \/ StartOpen \/ FullyOpen \/ StartClose \/ FullyClosed \/ Done

Spec == Init /\ [][Next]_vars

\* Opening only from closed; closing only from open; never both at once.
SafetyInvariant == ((state = "opening") => (prev_state = "closed")) /\ ((state = "closing") => (prev_state = "open")) /\ ~(state = "opening" /\ prev_state = "closing")

TypeOK == /\ state \in States
          /\ prev_state \in States
          /\ cycles \in 0..MaxCycles
          /\ SafetyInvariant
====
