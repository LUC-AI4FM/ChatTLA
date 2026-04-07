---- MODULE CpuQuota ----
(***************************************************************************)
(* Per-task CPU quota: each task may use at most Q ticks per period of P. *)
(* When the period elapses, all per-task usage counters reset.            *)
(* Safety: every task's per-period usage stays in 0..Q.                   *)
(***************************************************************************)
EXTENDS Naturals

CONSTANT N

ASSUME N \in 1..3

Tasks == 0..(N-1)
Q == 2  \* quota ticks per period
P == 3  \* period length in ticks

VARIABLES used, clock

vars == << used, clock >>

Init == /\ used  = [t \in Tasks |-> 0]
        /\ clock = 0

\* Schedule one tick of work to a task that still has quota and time in period.
Run(t) == /\ used[t] < Q
          /\ clock < P
          /\ used' = [used EXCEPT ![t] = @ + 1]
          /\ clock' = clock + 1

\* Idle tick (no task runs this tick) — period clock still advances.
Idle == /\ clock < P
        /\ clock' = clock + 1
        /\ UNCHANGED used

\* Period boundary: reset usage and clock.
Reset == /\ clock = P
         /\ used' = [t \in Tasks |-> 0]
         /\ clock' = 0

Next == (\E t \in Tasks : Run(t)) \/ Idle \/ Reset

Spec == Init /\ [][Next]_vars

QuotaInv == \A t \in Tasks : used[t] \in 0..Q

TypeOK == /\ clock \in 0..P /\ QuotaInv
====
