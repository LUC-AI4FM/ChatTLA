---- MODULE Quota ----
(***************************************************************************)
(* Per-tenant quota: each of T tenants is capped at C concurrent ops.     *)
(* Safety: per-tenant in-flight count never exceeds C.                    *)
(***************************************************************************)
EXTENDS Naturals

CONSTANT N

ASSUME N \in 1..3   \* number of tenants

Tenants == 0..(N-1)
C == 2  \* per-tenant quota

VARIABLE inflight

vars == << inflight >>

Init == inflight = [t \in Tenants |-> 0]

\* Tenant t starts a new operation, if under quota.
Start(t) == /\ inflight[t] < C
            /\ inflight' = [inflight EXCEPT ![t] = @ + 1]

\* Tenant t finishes an operation.
Finish(t) == /\ inflight[t] > 0
             /\ inflight' = [inflight EXCEPT ![t] = @ - 1]

Next == \E t \in Tenants : Start(t) \/ Finish(t)

Spec == Init /\ [][Next]_vars

QuotaInv == \A t \in Tenants : inflight[t] \in 0..C

TypeOK == QuotaInv
====
