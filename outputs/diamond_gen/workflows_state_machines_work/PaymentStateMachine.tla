---- MODULE PaymentStateMachine ----
(***************************************************************************)
(*  Payment processing state machine.                                      *)
(*    pending -> authorized -> captured -> refunded                        *)
(*    pending -> authorized -> voided                                      *)
(*  History flags ensure refund only after capture and void only when      *)
(*  capture has not happened.                                              *)
(***************************************************************************)
EXTENDS Naturals

VARIABLES status, ever_authorized, ever_captured, ever_voided

vars == << status, ever_authorized, ever_captured, ever_voided >>

States == {"pending", "authorized", "captured", "voided", "refunded"}

Init == /\ status           = "pending"
        /\ ever_authorized   = FALSE
        /\ ever_captured     = FALSE
        /\ ever_voided       = FALSE

Authorize == /\ status = "pending"
             /\ status' = "authorized"
             /\ ever_authorized' = TRUE
             /\ UNCHANGED << ever_captured, ever_voided >>

Capture == /\ status = "authorized"
           /\ status' = "captured"
           /\ ever_captured' = TRUE
           /\ UNCHANGED << ever_authorized, ever_voided >>

Void == /\ status = "authorized"
        /\ status' = "voided"
        /\ ever_voided' = TRUE
        /\ UNCHANGED << ever_authorized, ever_captured >>

Refund == /\ status = "captured"
          /\ status' = "refunded"
          /\ UNCHANGED << ever_authorized, ever_captured, ever_voided >>

Done == /\ status \in {"voided", "refunded"}
        /\ UNCHANGED vars

Next == \/ Authorize \/ Capture \/ Void \/ Refund \/ Done

Spec == Init /\ [][Next]_vars

\* Refunds require a prior capture; voids require capture never happened.
SafetyInvariant == ((status = "refunded") => ever_captured) /\ ((status = "voided") => (~ever_captured)) /\ ((status \in {"captured","refunded"}) => ever_authorized)

TypeOK == /\ status \in States
          /\ ever_authorized \in BOOLEAN
          /\ ever_captured   \in BOOLEAN
          /\ ever_voided     \in BOOLEAN
          /\ SafetyInvariant
====
