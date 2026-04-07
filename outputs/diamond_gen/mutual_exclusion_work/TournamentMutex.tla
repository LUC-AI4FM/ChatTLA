---- MODULE TournamentMutex ----
(***************************************************************************)
(* Tournament-tree mutual exclusion (Peterson 1981 / Yang-Anderson 1995).  *)
(* For 2^k processes a binary tree of 2-process Peterson nodes is built.   *)
(* A process traverses the tree from leaf to root, winning each node.      *)
(* Here we model k = 1: 2 processes, single root node.                     *)
(***************************************************************************)
EXTENDS Naturals

Procs == {0, 1}

VARIABLES pc, flag, turn

vars == << pc, flag, turn >>

\* The single tree node is the root; in a deeper tree there would be one
\* (flag, turn) pair per internal node.
Other(i) == 1 - i

Init == /\ pc   = [i \in Procs |-> "leaf"]
        /\ flag = [i \in Procs |-> FALSE]
        /\ turn = 0

\* Move from leaf to root by raising flag and ceding turn at the root node.
ClimbRoot(i) ==
    /\ pc[i] = "leaf"
    /\ flag' = [flag EXCEPT ![i] = TRUE]
    /\ turn' = Other(i)
    /\ pc'   = [pc EXCEPT ![i] = "wait_root"]

\* Win the root node: either the other isn't competing or it isn't its turn.
WinRoot(i) ==
    /\ pc[i] = "wait_root"
    /\ (flag[Other(i)] = FALSE \/ turn = i)
    /\ pc' = [pc EXCEPT ![i] = "cs"]
    /\ UNCHANGED << flag, turn >>

\* Release the root and return to leaf.
Release(i) ==
    /\ pc[i] = "cs"
    /\ flag' = [flag EXCEPT ![i] = FALSE]
    /\ pc'   = [pc EXCEPT ![i] = "leaf"]
    /\ UNCHANGED turn

Next == \E i \in Procs : ClimbRoot(i) \/ WinRoot(i) \/ Release(i)

Spec == Init /\ [][Next]_vars

TypeOK ==
    /\ pc   \in [Procs -> {"leaf","wait_root","cs"}]
    /\ flag \in [Procs -> BOOLEAN]
    /\ turn \in Procs
    /\ \A i, j \in Procs : (i # j /\ pc[i] = "cs") => pc[j] # "cs"
====
