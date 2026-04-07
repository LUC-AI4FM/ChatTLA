---- MODULE DeadlineScheduler ----
(***************************************************************************)
(* Earliest-deadline-first (EDF) scheduler over N tasks.  Each task has a *)
(* remaining deadline that decreases each tick.  At every step the       *)
(* scheduler runs the ready task with the smallest remaining deadline.   *)
(*                                                                         *)
(* Safety: the running task always has the minimum deadline among ready  *)
(* tasks.                                                                *)
(***************************************************************************)
EXTENDS Naturals

CONSTANT N

ASSUME N \in 2..3

Tasks == 0..(N-1)
NoTask == N
MaxDL == 3

VARIABLES dl, ready, running

vars == << dl, ready, running >>

Init == /\ dl      = [t \in Tasks |-> MaxDL]
        /\ ready   = Tasks
        /\ running = NoTask

\* Set of minimum-deadline ready tasks.
MinReadyDL ==
  IF ready = {} THEN MaxDL + 1
  ELSE CHOOSE m \in {dl[t] : t \in ready} :
         \A k \in {dl[t] : t \in ready} : m <= k

\* Pick a ready task whose deadline equals the minimum.
Dispatch == /\ running = NoTask
            /\ ready # {}
            /\ \E t \in ready :
                 /\ dl[t] = MinReadyDL
                 /\ running' = t
                 /\ ready'   = ready \ {t}
                 /\ UNCHANGED dl

\* Tick: every task's remaining deadline decreases by one (clamped at 0).
\* Aging the running task in lockstep keeps the EDF invariant intact.
Tick == /\ running # NoTask
        /\ dl' = [t \in Tasks |-> IF dl[t] > 0 THEN dl[t] - 1 ELSE 0]
        /\ UNCHANGED << ready, running >>

\* Running task completes and is reborn with a fresh deadline.
Complete == /\ running # NoTask
            /\ dl' = [dl EXCEPT ![running] = MaxDL]
            /\ ready' = ready \cup {running}
            /\ running' = NoTask

Next == Dispatch \/ Tick \/ Complete

Spec == Init /\ [][Next]_vars

\* Strong safety: when a task is running, no ready task has a smaller deadline.
EdfInv == (running = NoTask) \/ (\A t \in ready : dl[running] <= dl[t])

TypeOK == /\ running \in Tasks \cup {NoTask}
          /\ ready \subseteq Tasks
          /\ \A t \in Tasks : dl[t] \in 0..MaxDL
          /\ EdfInv
====
