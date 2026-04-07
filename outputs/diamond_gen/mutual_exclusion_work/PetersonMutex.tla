---- MODULE PetersonMutex ----
(***************************************************************************)
(* Peterson's algorithm for two processes (1981).                          *)
(* Each process i has a flag[i] announcing intent and a shared turn.       *)
(* Process i enters the critical section only when flag[j] = FALSE or      *)
(* turn = i.  Mutual exclusion is the canonical safety property.           *)
(***************************************************************************)
EXTENDS Naturals

Procs == {0, 1}

VARIABLES pc, flag, turn

vars == << pc, flag, turn >>

\* Other(i) is the unique competing process.
Other(i) == 1 - i

\* Init: both processes idle, neither flag set, turn arbitrary (= 0).
Init == /\ pc   = [i \in Procs |-> "ncs"]
        /\ flag = [i \in Procs |-> FALSE]
        /\ turn = 0

\* Step 1: leave the non-critical section, set own flag.
SetFlag(i) ==
    /\ pc[i] = "ncs"
    /\ pc'   = [pc   EXCEPT ![i] = "set"]
    /\ flag' = [flag EXCEPT ![i] = TRUE]
    /\ UNCHANGED turn

\* Step 2: yield turn to the other process.
SetTurn(i) ==
    /\ pc[i] = "set"
    /\ pc'   = [pc EXCEPT ![i] = "wait"]
    /\ turn' = Other(i)
    /\ UNCHANGED flag

\* Step 3: wait until other has not raised its flag, or it is our turn.
Enter(i) ==
    /\ pc[i] = "wait"
    /\ (flag[Other(i)] = FALSE \/ turn = i)
    /\ pc' = [pc EXCEPT ![i] = "cs"]
    /\ UNCHANGED << flag, turn >>

\* Leave the critical section and clear our flag.
Leave(i) ==
    /\ pc[i] = "cs"
    /\ pc'   = [pc   EXCEPT ![i] = "ncs"]
    /\ flag' = [flag EXCEPT ![i] = FALSE]
    /\ UNCHANGED turn

Next == \E i \in Procs :
            SetFlag(i) \/ SetTurn(i) \/ Enter(i) \/ Leave(i)

Spec == Init /\ [][Next]_vars

\* TypeOK conjoins the strong mutual-exclusion safety property so that the
\* Diamond mutation test sees an invariant which actually constrains states.
TypeOK ==
    /\ pc   \in [Procs -> {"ncs", "set", "wait", "cs"}]
    /\ flag \in [Procs -> BOOLEAN]
    /\ turn \in Procs
    /\ \A i, j \in Procs : (i # j /\ pc[i] = "cs") => pc[j] # "cs"
====
