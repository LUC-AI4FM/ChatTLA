---- MODULE DocumentApproval ----
(***************************************************************************)
(*  Document approval requiring sequential signatures from N approvers     *)
(*  in strict order. Track signature count; document is published when    *)
(*  signatures = N.                                                       *)
(***************************************************************************)
EXTENDS Naturals

CONSTANT NumApprovers

VARIABLES status, signatures, ever_drafted

vars == << status, signatures, ever_drafted >>

States == {"draft", "circulating", "published", "withdrawn"}

Init == /\ status       = "draft"
        /\ signatures   = 0
        /\ ever_drafted = TRUE

Submit == /\ status = "draft"
          /\ status' = "circulating"
          /\ UNCHANGED << signatures, ever_drafted >>

Sign == /\ status = "circulating"
        /\ signatures < NumApprovers
        /\ signatures' = signatures + 1
        /\ status' = IF signatures + 1 = NumApprovers THEN "published" ELSE "circulating"
        /\ UNCHANGED ever_drafted

Withdraw == /\ status \in {"draft", "circulating"}
            /\ status' = "withdrawn"
            /\ UNCHANGED << signatures, ever_drafted >>

Done == /\ status \in {"published", "withdrawn"}
        /\ UNCHANGED vars

Next == \/ Submit \/ Sign \/ Withdraw \/ Done

Spec == Init /\ [][Next]_vars

\* Published implies all approvers signed in order.
SafetyInvariant == ((status = "published") => (signatures = NumApprovers /\ ever_drafted)) /\ (signatures <= NumApprovers) /\ ((status = "draft") => signatures = 0)

TypeOK == /\ status \in States
          /\ signatures \in 0..NumApprovers
          /\ ever_drafted \in BOOLEAN
          /\ SafetyInvariant
====
