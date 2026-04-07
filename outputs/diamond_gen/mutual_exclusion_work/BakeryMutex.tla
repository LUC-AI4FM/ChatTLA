---- MODULE BakeryMutex ----
(***************************************************************************)
(* Lamport's Bakery Algorithm (1974).  Each process picks a number larger  *)
(* than any currently held; the smallest (process id breaking ties) wins.  *)
(* `choosing[i]` is set while picking the number to avoid races.           *)
(***************************************************************************)
EXTENDS Naturals

N == 2
Procs == 1..N
MaxNum == 3

VARIABLES pc, choosing, number

vars == << pc, choosing, number >>

\* Lexicographic compare on (number[i], i).
LessEq(a, i, b, j) ==
    \/ a < b
    \/ (a = b /\ i <= j)

Init == /\ pc       = [i \in Procs |-> "ncs"]
        /\ choosing = [i \in Procs |-> FALSE]
        /\ number   = [i \in Procs |-> 0]

\* Begin choosing a number.
StartChoose(i) ==
    /\ pc[i] = "ncs"
    /\ choosing' = [choosing EXCEPT ![i] = TRUE]
    /\ pc'       = [pc       EXCEPT ![i] = "choose"]
    /\ UNCHANGED number

\* Pick number = 1 + max(others), bounded by MaxNum.
Pick(i) ==
    LET others == {number[j] : j \in Procs \ {i}}
        mx     == IF others = {} THEN 0
                  ELSE CHOOSE x \in others : \A y \in others : y <= x
    IN  /\ pc[i] = "choose"
        /\ mx + 1 <= MaxNum
        /\ number'   = [number   EXCEPT ![i] = mx + 1]
        /\ choosing' = [choosing EXCEPT ![i] = FALSE]
        /\ pc'       = [pc       EXCEPT ![i] = "wait"]

\* Enter only when no other is choosing and we have the smallest ticket.
Enter(i) ==
    /\ pc[i] = "wait"
    /\ \A j \in Procs \ {i} :
         /\ choosing[j] = FALSE
         /\ (number[j] = 0 \/ LessEq(number[i], i, number[j], j))
    /\ pc' = [pc EXCEPT ![i] = "cs"]
    /\ UNCHANGED << choosing, number >>

Leave(i) ==
    /\ pc[i] = "cs"
    /\ number' = [number EXCEPT ![i] = 0]
    /\ pc'     = [pc     EXCEPT ![i] = "ncs"]
    /\ UNCHANGED choosing

\* Self-loop disjunct so model checking never deadlocks at the number bound.
Idle == UNCHANGED vars

Next == \/ \E i \in Procs :
            StartChoose(i) \/ Pick(i) \/ Enter(i) \/ Leave(i)
        \/ Idle

Spec == Init /\ [][Next]_vars

TypeOK ==
    /\ pc       \in [Procs -> {"ncs","choose","wait","cs"}]
    /\ choosing \in [Procs -> BOOLEAN]
    /\ number   \in [Procs -> 0..MaxNum]
    /\ \A i, j \in Procs : (i # j /\ pc[i] = "cs") => pc[j] # "cs"

\* Bound numbers so TLC explores a finite state space.
NumberBound == \A i \in Procs : number[i] <= MaxNum
====
SPECIFICATION Spec
INVARIANT TypeOK
CONSTRAINT NumberBound
