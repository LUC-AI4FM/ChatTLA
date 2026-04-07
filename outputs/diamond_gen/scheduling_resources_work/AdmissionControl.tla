---- MODULE AdmissionControl ----
(***************************************************************************)
(* Concurrency-based admission control: a request is admitted only when  *)
(* the active count is strictly below the configured capacity C.         *)
(*                                                                         *)
(* Safety: active never exceeds C.                                        *)
(***************************************************************************)
EXTENDS Naturals

C == 3  \* capacity (max concurrent active requests)

VARIABLES active, rejected

vars == << active, rejected >>

Init == /\ active = 0
        /\ rejected = 0

\* Admit a request.
Admit == /\ active < C
         /\ active' = active + 1
         /\ UNCHANGED rejected

\* Reject (full).
Reject == /\ active = C
          /\ rejected < C   \* finite bound
          /\ rejected' = rejected + 1
          /\ UNCHANGED active

\* Complete an active request.
Complete == /\ active > 0
            /\ active' = active - 1
            /\ UNCHANGED rejected

\* Reset reject counter to keep the state space small.
ResetRejected == /\ rejected > 0
                 /\ rejected' = 0
                 /\ UNCHANGED active

Next == Admit \/ Reject \/ Complete \/ ResetRejected

Spec == Init /\ [][Next]_vars

CapacityInv == active \in 0..C /\ rejected \in 0..C

TypeOK == CapacityInv
====
