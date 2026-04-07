---- MODULE ContentModeration ----
(***************************************************************************)
(*  Content moderation workflow.                                           *)
(*    submitted -> auto_review -> human_review -> approved | rejected      *)
(*  Items only become published if they were approved.                     *)
(***************************************************************************)
EXTENDS Naturals

VARIABLES status, ever_auto_passed, ever_human_passed, published

vars == << status, ever_auto_passed, ever_human_passed, published >>

States == {"submitted", "auto_review", "human_review", "approved", "rejected", "published"}

Init == /\ status            = "submitted"
        /\ ever_auto_passed   = FALSE
        /\ ever_human_passed  = FALSE
        /\ published          = FALSE

StartAuto == /\ status = "submitted"
             /\ status' = "auto_review"
             /\ UNCHANGED << ever_auto_passed, ever_human_passed, published >>

AutoPass == /\ status = "auto_review"
            /\ status' = "human_review"
            /\ ever_auto_passed' = TRUE
            /\ UNCHANGED << ever_human_passed, published >>

AutoReject == /\ status = "auto_review"
              /\ status' = "rejected"
              /\ UNCHANGED << ever_auto_passed, ever_human_passed, published >>

HumanApprove == /\ status = "human_review"
                /\ status' = "approved"
                /\ ever_human_passed' = TRUE
                /\ UNCHANGED << ever_auto_passed, published >>

HumanReject == /\ status = "human_review"
               /\ status' = "rejected"
               /\ UNCHANGED << ever_auto_passed, ever_human_passed, published >>

Publish == /\ status = "approved"
           /\ status' = "published"
           /\ published' = TRUE
           /\ UNCHANGED << ever_auto_passed, ever_human_passed >>

Done == /\ status \in {"published", "rejected"}
        /\ UNCHANGED vars

Next == \/ StartAuto \/ AutoPass \/ AutoReject \/ HumanApprove \/ HumanReject \/ Publish \/ Done

Spec == Init /\ [][Next]_vars

\* Published iff content was approved (which required passing both reviews).
SafetyInvariant == (published => (ever_auto_passed /\ ever_human_passed)) /\ ((status = "published") => published) /\ ((status = "approved") => (ever_auto_passed /\ ever_human_passed))

TypeOK == /\ status \in States
          /\ ever_auto_passed \in BOOLEAN
          /\ ever_human_passed \in BOOLEAN
          /\ published \in BOOLEAN
          /\ SafetyInvariant
====
