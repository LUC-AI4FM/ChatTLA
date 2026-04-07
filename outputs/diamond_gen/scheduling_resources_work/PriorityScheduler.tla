---- MODULE PriorityScheduler ----
(***************************************************************************)
(* Preemptive priority scheduler over N tasks with priorities in 1..3.     *)
(* A higher-priority ready task always preempts a lower-priority running.  *)
(* Safety: the running task has the highest priority among all ready.      *)
(***************************************************************************)
EXTENDS Naturals, FiniteSets

CONSTANT N

ASSUME N \in 1..4

\* Fixed priority assignment for the model: task i has priority (i mod 3) + 1.
Prio(i) == (i % 3) + 1

Tasks == 0..(N-1)
NoTask == N

VARIABLES running, ready

vars == << running, ready >>

Init == /\ running = NoTask
        /\ ready = Tasks  \* all tasks start ready

\* Maximum priority among the ready set (0 if empty).
MaxReadyPrio == IF ready = {} THEN 0
                ELSE CHOOSE p \in {Prio(t) : t \in ready} :
                       \A q \in {Prio(t) : t \in ready} : q <= p

\* Dispatch a highest-priority ready task to run.
Dispatch == /\ running = NoTask
            /\ ready # {}
            /\ \E t \in ready :
                 /\ Prio(t) = MaxReadyPrio
                 /\ running' = t
                 /\ ready' = ready \ {t}

\* Running task completes.
Complete == /\ running # NoTask
            /\ running' = NoTask
            /\ UNCHANGED ready

\* New work arrives: a previously completed task becomes ready.
\* If its priority strictly exceeds the running task's, preempt immediately
\* (the running task is pushed back to ready); otherwise just enqueue.
Arrive == /\ \E t \in Tasks :
              /\ t \notin ready
              /\ t # running
              /\ \/ /\ running = NoTask
                    /\ ready' = ready \cup {t}
                    /\ UNCHANGED running
                 \/ /\ running # NoTask
                    /\ Prio(t) <= Prio(running)
                    /\ ready' = ready \cup {t}
                    /\ UNCHANGED running
                 \/ /\ running # NoTask
                    /\ Prio(t) > Prio(running)
                    /\ ready' = (ready \cup {running}) \ {t}
                    /\ running' = t

Next == Dispatch \/ Complete \/ Arrive

Spec == Init /\ [][Next]_vars

\* Strong safety: running task's priority is >= every ready task's priority.
PriorityInv == (running = NoTask) \/ (\A t \in ready : Prio(running) >= Prio(t))

TypeOK == /\ ready \subseteq Tasks /\ running \in Tasks \cup {NoTask} /\ PriorityInv
====
