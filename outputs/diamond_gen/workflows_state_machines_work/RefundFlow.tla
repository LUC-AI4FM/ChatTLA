---- MODULE RefundFlow ----
(***************************************************************************)
(*  Refund workflow with bounded partial refunds.                          *)
(*  An order is captured for an Original amount; refunds may be issued     *)
(*  one at a time, and the running total may never exceed Original.        *)
(***************************************************************************)
EXTENDS Naturals

CONSTANT Original

VARIABLES status, refunded_total, ever_captured

vars == << status, refunded_total, ever_captured >>

States == {"pending", "captured", "partially_refunded", "fully_refunded"}

Init == /\ status         = "pending"
        /\ refunded_total = 0
        /\ ever_captured  = FALSE

Capture == /\ status = "pending"
           /\ status' = "captured"
           /\ ever_captured' = TRUE
           /\ UNCHANGED refunded_total

PartialRefund == /\ status \in {"captured", "partially_refunded"}
                 /\ refunded_total < Original
                 /\ refunded_total' = refunded_total + 1
                 /\ status' = IF refunded_total + 1 = Original THEN "fully_refunded" ELSE "partially_refunded"
                 /\ UNCHANGED ever_captured

Done == /\ status = "fully_refunded"
        /\ UNCHANGED vars

Next == \/ Capture \/ PartialRefund \/ Done

Spec == Init /\ [][Next]_vars

\* Refunds bounded by original amount; refunds only after capture.
SafetyInvariant == (refunded_total <= Original) /\ ((refunded_total > 0) => ever_captured) /\ ((status \in {"partially_refunded", "fully_refunded"}) => ever_captured)

TypeOK == /\ status \in States
          /\ refunded_total \in 0..Original
          /\ ever_captured \in BOOLEAN
          /\ SafetyInvariant
====
