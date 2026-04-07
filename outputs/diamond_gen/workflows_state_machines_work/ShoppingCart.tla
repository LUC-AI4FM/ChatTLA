---- MODULE ShoppingCart ----
(***************************************************************************)
(*  Shopping cart with bounded item count and a one-shot checkout.         *)
(***************************************************************************)
EXTENDS Naturals

CONSTANT MaxItems

VARIABLES status, items, ever_added, order_count

vars == << status, items, ever_added, order_count >>

States == {"shopping", "checked_out", "abandoned"}

Init == /\ status      = "shopping"
        /\ items       = 0
        /\ ever_added  = FALSE
        /\ order_count = 0

AddItem == /\ status = "shopping"
           /\ items < MaxItems
           /\ items' = items + 1
           /\ ever_added' = TRUE
           /\ UNCHANGED << status, order_count >>

RemoveItem == /\ status = "shopping"
              /\ items > 0
              /\ items' = items - 1
              /\ UNCHANGED << status, ever_added, order_count >>

Checkout == /\ status = "shopping"
            /\ items > 0
            /\ status' = "checked_out"
            /\ order_count' = order_count + 1
            /\ UNCHANGED << items, ever_added >>

Abandon == /\ status = "shopping"
           /\ status' = "abandoned"
           /\ UNCHANGED << items, ever_added, order_count >>

Done == /\ status \in {"checked_out", "abandoned"}
        /\ UNCHANGED vars

Next == \/ AddItem \/ RemoveItem \/ Checkout \/ Abandon \/ Done

Spec == Init /\ [][Next]_vars

\* Cart bounded; checkout requires items and produces exactly one order.
SafetyInvariant == (items <= MaxItems) /\ ((status = "checked_out") => (ever_added /\ order_count = 1)) /\ (order_count <= 1) /\ ((order_count = 1) => (status = "checked_out"))

TypeOK == /\ status \in States
          /\ items \in 0..MaxItems
          /\ ever_added \in BOOLEAN
          /\ order_count \in 0..1
          /\ SafetyInvariant
====
