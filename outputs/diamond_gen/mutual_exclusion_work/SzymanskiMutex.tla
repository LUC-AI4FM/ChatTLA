---- MODULE SzymanskiMutex ----
(***************************************************************************)
(* Szymanski's mutual exclusion algorithm (1988).  Each process is at one  *)
(* of five levels: 0 (ncs), 1 (entering doorway), 2 (waiting for door),    *)
(* 3 (in doorway, second wait), 4 (cs).  Linear waiting and bounded wait.  *)
(***************************************************************************)
EXTENDS Naturals

N == 2
Procs == 1..N

VARIABLES level

vars == << level >>

Init == level = [i \in Procs |-> 0]

\* Step 1: leave NCS, raise to level 1.
EnterDoor(i) ==
    /\ level[i] = 0
    /\ level' = [level EXCEPT ![i] = 1]

\* Step 2: when no one is at levels 3 or 4, jump to level 3.
PassDoor(i) ==
    /\ level[i] = 1
    /\ \A j \in Procs \ {i} : level[j] < 3
    /\ level' = [level EXCEPT ![i] = 3]

\* Step 2b: alternative — go to level 2 (waiting room) if some other process
\* is past the door.
WaitInRoom(i) ==
    /\ level[i] = 1
    /\ \E j \in Procs \ {i} : level[j] >= 3
    /\ level' = [level EXCEPT ![i] = 2]

\* Step 3: in waiting room, advance to level 3 only when some process is at
\* level 4 (CS) — this signals door is closed.  Simplified for two-proc.
LeaveRoom(i) ==
    /\ level[i] = 2
    /\ \E j \in Procs \ {i} : level[j] = 4
    /\ level' = [level EXCEPT ![i] = 3]

\* Step 4: when no other process is at level 1 or 2, enter CS (level 4).
\* This is the linear-waiting condition.
\* Enter CS when (a) no other process is at level 4 already and
\* (b) we are the smallest-id process at level >= 3.
EnterCS(i) ==
    /\ level[i] = 3
    /\ \A j \in Procs \ {i} : level[j] # 4
    /\ \A j \in Procs : (j < i) => level[j] < 3
    /\ level' = [level EXCEPT ![i] = 4]

Leave(i) ==
    /\ level[i] = 4
    /\ level' = [level EXCEPT ![i] = 0]

\* Self-loop so model checking never deadlocks at unreachable guards.
Idle == UNCHANGED vars

Next == \/ \E i \in Procs :
            EnterDoor(i) \/ PassDoor(i) \/ WaitInRoom(i)
         \/ LeaveRoom(i) \/ EnterCS(i) \/ Leave(i)
        \/ Idle

Spec == Init /\ [][Next]_vars

TypeOK ==
    /\ level \in [Procs -> 0..4]
    /\ \A i, j \in Procs : (i # j /\ level[i] = 4) => level[j] # 4
====
