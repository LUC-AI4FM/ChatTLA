---- MODULE MultilevelFeedbackQueue ----
(***************************************************************************)
(* MLFQ scheduler with 3 priority levels (0 = highest, 2 = lowest).        *)
(* Each task starts at level 0; on quantum exhaustion it is demoted        *)
(* one level (capped at level 2). On a (modeled) I/O block it stays.       *)
(* Safety: every task lives in exactly one level; level only decreases.    *)
(***************************************************************************)
EXTENDS Naturals

CONSTANT N

ASSUME N \in 1..3

Tasks == 0..(N-1)
Levels == 0..2
Quantum == 2  \* ticks before demotion

VARIABLES level, used, running

vars == << level, used, running >>

NoTask == N

Init == /\ level = [t \in Tasks |-> 0]
        /\ used  = [t \in Tasks |-> 0]
        /\ running = NoTask

\* Highest priority level that contains at least one ready task
\* (ready = not currently running and level value).
LowestLevelWithReady ==
  CHOOSE L \in Levels :
    /\ \E t \in Tasks : t # running /\ level[t] = L
    /\ \A K \in 0..(L-1) : ~ (\E t \in Tasks : t # running /\ level[t] = K)

ReadyExists == \E t \in Tasks : t # running

\* Pick a task at the highest non-empty level.
Schedule == /\ running = NoTask
            /\ ReadyExists
            /\ \E t \in Tasks :
                 /\ level[t] = LowestLevelWithReady
                 /\ running' = t
                 /\ UNCHANGED << level, used >>

\* Use one tick of quantum without exhausting it.
Tick == /\ running # NoTask
        /\ used[running] + 1 < Quantum
        /\ used' = [used EXCEPT ![running] = @ + 1]
        /\ UNCHANGED << level, running >>

\* Quantum exhausted: demote (unless at lowest level) and yield.
Demote == /\ running # NoTask
          /\ used[running] + 1 >= Quantum
          /\ level' = [level EXCEPT ![running] =
                         IF @ < 2 THEN @ + 1 ELSE @]
          /\ used' = [used EXCEPT ![running] = 0]
          /\ running' = NoTask

\* Modeled I/O: task voluntarily yields without using full quantum.
IOBlock == /\ running # NoTask
           /\ used' = [used EXCEPT ![running] = 0]
           /\ running' = NoTask
           /\ UNCHANGED level

Next == Schedule \/ Tick \/ Demote \/ IOBlock

Spec == Init /\ [][Next]_vars

\* Strong safety: levels are valid AND used quantum never reaches Quantum
\* (because Demote fires at the boundary, resetting it).
LevelInv == \A t \in Tasks : level[t] \in Levels /\ used[t] \in 0..(Quantum-1)

TypeOK == /\ running \in Tasks \cup {NoTask} /\ LevelInv
====
