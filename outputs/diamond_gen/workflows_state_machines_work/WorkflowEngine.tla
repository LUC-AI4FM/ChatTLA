---- MODULE WorkflowEngine ----
(***************************************************************************)
(*  Generic three-state workflow engine.                                   *)
(*    pending -> active -> done                                            *)
(*  Transitions only along the allowed edges; track history flags so the   *)
(*  monotone phase ordering is encoded as a state predicate.               *)
(***************************************************************************)
EXTENDS Naturals

VARIABLES state, ever_active, ever_done

vars == << state, ever_active, ever_done >>

States == {"pending", "active", "done"}

Init == /\ state       = "pending"
        /\ ever_active = FALSE
        /\ ever_done   = FALSE

Activate == /\ state = "pending"
            /\ state' = "active"
            /\ ever_active' = TRUE
            /\ UNCHANGED ever_done

Complete == /\ state = "active"
            /\ state' = "done"
            /\ ever_done' = TRUE
            /\ UNCHANGED ever_active

Stay == /\ state = "done"
        /\ UNCHANGED vars

Next == \/ Activate \/ Complete \/ Stay

Spec == Init /\ [][Next]_vars

\* The engine never skips phases: done implies it was active first.
SafetyInvariant == ((state = "done") => (ever_active /\ ever_done)) /\ ((state = "active") => ever_active) /\ ((state = "pending") => (~ever_active /\ ~ever_done))

TypeOK == /\ state \in States
          /\ ever_active \in BOOLEAN
          /\ ever_done \in BOOLEAN
          /\ SafetyInvariant
====
