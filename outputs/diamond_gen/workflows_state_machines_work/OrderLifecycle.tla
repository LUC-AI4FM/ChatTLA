---- MODULE OrderLifecycle ----
(***************************************************************************)
(*  Order workflow lifecycle:                                              *)
(*    created -> paid -> shipped -> delivered                              *)
(*  Cancellation is allowed only before shipping.                          *)
(*  History flags (ever_paid, ever_shipped) make the safety property       *)
(*  "delivered implies previously shipped and paid" expressible as a        *)
(*  state-predicate invariant TLC can check directly.                      *)
(***************************************************************************)
EXTENDS Naturals

VARIABLES status, ever_paid, ever_shipped, ever_cancelled

vars == << status, ever_paid, ever_shipped, ever_cancelled >>

States == {"created", "paid", "shipped", "delivered", "cancelled"}

Init == /\ status        = "created"
        /\ ever_paid      = FALSE
        /\ ever_shipped   = FALSE
        /\ ever_cancelled = FALSE

Pay == /\ status = "created"
       /\ status' = "paid"
       /\ ever_paid' = TRUE
       /\ UNCHANGED << ever_shipped, ever_cancelled >>

Ship == /\ status = "paid"
        /\ status' = "shipped"
        /\ ever_shipped' = TRUE
        /\ UNCHANGED << ever_paid, ever_cancelled >>

Deliver == /\ status = "shipped"
           /\ status' = "delivered"
           /\ UNCHANGED << ever_paid, ever_shipped, ever_cancelled >>

Cancel == /\ status \in {"created", "paid"}
          /\ status' = "cancelled"
          /\ ever_cancelled' = TRUE
          /\ UNCHANGED << ever_paid, ever_shipped >>

\* Explicit terminal stutter so TLC sees the workflow's accepting end states
\* as reachable rather than as deadlocks.
Done == /\ status \in {"delivered", "cancelled"}
        /\ UNCHANGED vars

Next == \/ Pay \/ Ship \/ Deliver \/ Cancel \/ Done

Spec == Init /\ [][Next]_vars

\* Strong safety: a delivered order has been both paid and shipped along
\* its history; a cancelled order was never shipped.
SafetyInvariant == ((status = "delivered") => (ever_paid /\ ever_shipped)) /\ ((status = "cancelled") => (~ever_shipped))

TypeOK == /\ status \in States
          /\ ever_paid      \in BOOLEAN
          /\ ever_shipped   \in BOOLEAN
          /\ ever_cancelled \in BOOLEAN
          /\ SafetyInvariant
====
