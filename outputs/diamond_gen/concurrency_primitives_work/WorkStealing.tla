---- MODULE WorkStealing ----
EXTENDS Naturals

CONSTANT InitTasks

\* Two workers, A and B, each with a local task count and processed count.
\* localA / localB : tasks in worker's deque
\* doneA  / doneB  : tasks the worker has completed
VARIABLES localA, localB, doneA, doneB

vars == << localA, localB, doneA, doneB >>

Init == /\ localA = InitTasks
        /\ localB = 0
        /\ doneA  = 0
        /\ doneB  = 0

\* A processes one of its own tasks (pop from head).
ProcessA == /\ localA > 0
            /\ localA' = localA - 1
            /\ doneA'  = doneA + 1
            /\ UNCHANGED << localB, doneB >>

ProcessB == /\ localB > 0
            /\ localB' = localB - 1
            /\ doneB'  = doneB + 1
            /\ UNCHANGED << localA, doneA >>

\* B steals one task from A's tail when B is idle and A has > 1 task.
StealAfromB == /\ localB = 0
               /\ localA > 1
               /\ localA' = localA - 1
               /\ localB' = 1
               /\ UNCHANGED << doneA, doneB >>

StealBfromA == /\ localA = 0
               /\ localB > 1
               /\ localB' = localB - 1
               /\ localA' = 1
               /\ UNCHANGED << doneA, doneB >>

\* Reset to allow continued exploration once everything is processed.
Restart == /\ localA = 0
           /\ localB = 0
           /\ doneA + doneB = InitTasks
           /\ localA' = InitTasks
           /\ localB' = 0
           /\ doneA'  = 0
           /\ doneB'  = 0

Next == \/ ProcessA
        \/ ProcessB
        \/ StealAfromB
        \/ StealBfromA
        \/ Restart

Spec == Init /\ [][Next]_vars

\* Conservation: total tasks (queued + done) is always exactly InitTasks.
Conservation == localA + localB + doneA + doneB = InitTasks

TypeOK == /\ localA \in 0..InitTasks
          /\ localB \in 0..InitTasks
          /\ doneA  \in 0..InitTasks
          /\ doneB  \in 0..InitTasks
          /\ Conservation
====
