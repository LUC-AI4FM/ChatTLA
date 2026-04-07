---- MODULE MergeRequest ----
(***************************************************************************)
(*  Merge-request workflow.                                                *)
(*    draft -> ready -> approved -> merged                                  *)
(*  Merging requires at least RequiredApprovals approvers (in {0..N}).     *)
(***************************************************************************)
EXTENDS Naturals

CONSTANTS NumReviewers, RequiredApprovals

VARIABLES status, approver_count, ever_ready

vars == << status, approver_count, ever_ready >>

States == {"draft", "ready", "approved", "merged", "closed"}

Init == /\ status         = "draft"
        /\ approver_count = 0
        /\ ever_ready     = FALSE

MarkReady == /\ status = "draft"
             /\ status' = "ready"
             /\ ever_ready' = TRUE
             /\ UNCHANGED approver_count

Approve == /\ status = "ready"
           /\ approver_count < NumReviewers
           /\ approver_count' = approver_count + 1
           /\ status' = IF approver_count + 1 >= RequiredApprovals THEN "approved" ELSE "ready"
           /\ UNCHANGED ever_ready

Merge == /\ status = "approved"
         /\ approver_count >= RequiredApprovals
         /\ status' = "merged"
         /\ UNCHANGED << approver_count, ever_ready >>

Close == /\ status \in {"draft", "ready"}
         /\ status' = "closed"
         /\ UNCHANGED << approver_count, ever_ready >>

Done == /\ status \in {"merged", "closed"}
        /\ UNCHANGED vars

Next == \/ MarkReady \/ Approve \/ Merge \/ Close \/ Done

Spec == Init /\ [][Next]_vars

\* Merged implies the required number of approvals were collected and the
\* MR was promoted out of draft.
SafetyInvariant == ((status = "merged") => (approver_count >= RequiredApprovals /\ ever_ready)) /\ ((status = "approved") => approver_count >= RequiredApprovals) /\ (approver_count <= NumReviewers)

TypeOK == /\ status \in States
          /\ approver_count \in 0..NumReviewers
          /\ ever_ready \in BOOLEAN
          /\ SafetyInvariant
====
