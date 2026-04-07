---- MODULE TicketLifecycle ----
(***************************************************************************)
(*  Support ticket lifecycle.                                              *)
(*    open -> in_progress -> resolved -> closed                            *)
(*  Closed tickets may be reopened (back to open).                         *)
(***************************************************************************)
EXTENDS Naturals

CONSTANT MaxReopens

VARIABLES status, reopens, ever_in_progress, ever_resolved

vars == << status, reopens, ever_in_progress, ever_resolved >>

States == {"open", "in_progress", "resolved", "closed"}

Init == /\ status            = "open"
        /\ reopens            = 0
        /\ ever_in_progress   = FALSE
        /\ ever_resolved      = FALSE

Start == /\ status = "open"
         /\ status' = "in_progress"
         /\ ever_in_progress' = TRUE
         /\ UNCHANGED << reopens, ever_resolved >>

Resolve == /\ status = "in_progress"
           /\ status' = "resolved"
           /\ ever_resolved' = TRUE
           /\ UNCHANGED << reopens, ever_in_progress >>

Close == /\ status = "resolved"
         /\ status' = "closed"
         /\ UNCHANGED << reopens, ever_in_progress, ever_resolved >>

Reopen == /\ status = "closed"
          /\ reopens < MaxReopens
          /\ status' = "open"
          /\ reopens' = reopens + 1
          /\ UNCHANGED << ever_in_progress, ever_resolved >>

Done == /\ status = "closed"
        /\ reopens = MaxReopens
        /\ UNCHANGED vars

Next == \/ Start \/ Resolve \/ Close \/ Reopen \/ Done

Spec == Init /\ [][Next]_vars

\* Resolved/closed status implies ticket was previously worked on.
SafetyInvariant == ((status = "resolved") => ever_in_progress) /\ ((status = "closed") => (ever_in_progress /\ ever_resolved)) /\ (reopens <= MaxReopens)

TypeOK == /\ status \in States
          /\ reopens \in 0..MaxReopens
          /\ ever_in_progress \in BOOLEAN
          /\ ever_resolved \in BOOLEAN
          /\ SafetyInvariant
====
