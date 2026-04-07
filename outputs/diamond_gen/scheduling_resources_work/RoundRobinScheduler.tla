---- MODULE RoundRobinScheduler ----
(***************************************************************************)
(* Round-robin scheduler over N tasks sharing a single CPU.                *)
(* Tasks rotate strictly by index (0,1,2,0,1,2,...).                       *)
(* Safety: at most one running task; rotation order preserved.             *)
(***************************************************************************)
EXTENDS Naturals

CONSTANT N

ASSUME N \in 1..5

VARIABLES running, turn

vars == << running, turn >>

NoTask == N  \* sentinel value not in 0..N-1

Init == /\ running = NoTask
        /\ turn = 0

\* Schedule the task whose turn it currently is.
Schedule == /\ running = NoTask
            /\ running' = turn
            /\ UNCHANGED turn

\* Preempt the running task and rotate to the next.
Preempt == /\ running # NoTask
           /\ running' = NoTask
           /\ turn' = (turn + 1) % N

Next == Schedule \/ Preempt

Spec == Init /\ [][Next]_vars

\* Strong safety: a running task must equal the current turn (rotation respected).
RotationInv == (running = NoTask) \/ (running = turn)

TypeOK == /\ turn \in 0..(N-1) /\ running \in (0..(N-1)) \cup {NoTask} /\ RotationInv
====
