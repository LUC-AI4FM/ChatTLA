---- MODULE RWBakery ----
(***************************************************************************)
(* A reader-writer variant of Lamport's Bakery.  Both readers and writers  *)
(* draw a ticket.  A writer enters when its ticket is the smallest among   *)
(* all waiters AND no readers are active.  A reader enters when no writer  *)
(* with a smaller ticket is waiting or active.  Multiple readers may be    *)
(* concurrent; writers are exclusive against everyone.                     *)
(***************************************************************************)
EXTENDS Naturals

N == 2
Procs == 1..N
MaxNum == 3

VARIABLES pc, role, number

vars == << pc, role, number >>

LessEq(a, i, b, j) == \/ a < b
                     \/ (a = b /\ i <= j)

Init == /\ pc     = [i \in Procs |-> "ncs"]
        /\ role   = [i \in Procs |-> "reader"]
        /\ number = [i \in Procs |-> 0]

\* Pick a role and a ticket = 1 + max(others).
Pick(i, r) ==
    LET others == {number[j] : j \in Procs \ {i}}
        mx     == IF others = {} THEN 0
                  ELSE CHOOSE x \in others : \A y \in others : y <= x
    IN  /\ pc[i] = "ncs"
        /\ mx + 1 <= MaxNum
        /\ number' = [number EXCEPT ![i] = mx + 1]
        /\ role'   = [role   EXCEPT ![i] = r]
        /\ pc'     = [pc     EXCEPT ![i] = "wait"]

\* Reader enters: no writer holds a smaller-or-equal ticket.
EnterReader(i) ==
    /\ pc[i] = "wait"
    /\ role[i] = "reader"
    /\ \A j \in Procs \ {i} :
         (role[j] = "writer" /\ pc[j] \in {"wait","cs"}) =>
            LessEq(number[i], i, number[j], j)
    /\ pc' = [pc EXCEPT ![i] = "cs"]
    /\ UNCHANGED << role, number >>

\* Writer enters: its ticket is smallest AND no other process is in cs.
EnterWriter(i) ==
    /\ pc[i] = "wait"
    /\ role[i] = "writer"
    /\ \A j \in Procs \ {i} :
         /\ pc[j] # "cs"
         /\ (pc[j] = "wait" => LessEq(number[i], i, number[j], j))
    /\ pc' = [pc EXCEPT ![i] = "cs"]
    /\ UNCHANGED << role, number >>

Leave(i) ==
    /\ pc[i] = "cs"
    /\ number' = [number EXCEPT ![i] = 0]
    /\ pc'     = [pc     EXCEPT ![i] = "ncs"]
    /\ UNCHANGED role

Idle == UNCHANGED vars

Next == \/ \E i \in Procs, r \in {"reader","writer"} : Pick(i, r)
        \/ \E i \in Procs : EnterReader(i) \/ EnterWriter(i) \/ Leave(i)
        \/ Idle

Spec == Init /\ [][Next]_vars

\* Safety: a writer in CS excludes all other processes; readers may overlap.
TypeOK ==
    /\ pc     \in [Procs -> {"ncs","wait","cs"}]
    /\ role   \in [Procs -> {"reader","writer"}]
    /\ number \in [Procs -> 0..MaxNum]
    /\ \A i, j \in Procs :
         (i # j /\ pc[i] = "cs" /\ pc[j] = "cs") =>
            (role[i] = "reader" /\ role[j] = "reader")
====
