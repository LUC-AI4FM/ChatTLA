---- MODULE FastMutex ----
(***************************************************************************)
(* Lamport's Fast Mutex (1987).  Optimised for the uncontended case:       *)
(* in the absence of contention only five memory accesses are needed.      *)
(* Variables X and Y are shared; b[i] flags whether process i is trying.   *)
(***************************************************************************)
EXTENDS Naturals

N == 2
Procs == 1..N
NoProc == 0

VARIABLES pc, b, x, y

vars == << pc, b, x, y >>

Init == /\ pc = [i \in Procs |-> "ncs"]
        /\ b  = [i \in Procs |-> FALSE]
        /\ x  = NoProc
        /\ y  = NoProc

\* Step 1: announce intent.
Start(i) ==
    /\ pc[i] = "ncs"
    /\ b' = [b EXCEPT ![i] = TRUE]
    /\ pc' = [pc EXCEPT ![i] = "set_x"]
    /\ UNCHANGED << x, y >>

\* Step 2: x := i.
SetX(i) ==
    /\ pc[i] = "set_x"
    /\ x' = i
    /\ pc' = [pc EXCEPT ![i] = "check_y"]
    /\ UNCHANGED << b, y >>

\* Step 3: if y # NoProc, restart from the top (clear b, retry).
CheckY(i) ==
    /\ pc[i] = "check_y"
    /\ IF y # NoProc
         THEN /\ b' = [b EXCEPT ![i] = FALSE]
              /\ pc' = [pc EXCEPT ![i] = "ncs"]
              /\ UNCHANGED << x, y >>
         ELSE /\ pc' = [pc EXCEPT ![i] = "set_y"]
              /\ UNCHANGED << b, x, y >>

\* Step 4: y := i.
SetY(i) ==
    /\ pc[i] = "set_y"
    /\ y' = i
    /\ pc' = [pc EXCEPT ![i] = "check_x"]
    /\ UNCHANGED << b, x >>

\* Step 5: re-read x.  If x = i, fast-path enter.  Else slow-path wait.
CheckX(i) ==
    /\ pc[i] = "check_x"
    /\ IF x = i
         THEN /\ pc' = [pc EXCEPT ![i] = "cs"]
              /\ UNCHANGED << b, x, y >>
         ELSE /\ pc' = [pc EXCEPT ![i] = "wait_b"]
              /\ b' = [b EXCEPT ![i] = FALSE]
              /\ UNCHANGED << x, y >>

\* Slow path: wait for all other b[j] = FALSE, then check y = i.
WaitB(i) ==
    /\ pc[i] = "wait_b"
    /\ \A j \in Procs \ {i} : b[j] = FALSE
    /\ IF y = i
         THEN /\ pc' = [pc EXCEPT ![i] = "cs"]
              /\ UNCHANGED << b, x, y >>
         ELSE /\ pc' = [pc EXCEPT ![i] = "ncs"]
              /\ UNCHANGED << b, x, y >>

Leave(i) ==
    /\ pc[i] = "cs"
    /\ y' = NoProc
    /\ b' = [b EXCEPT ![i] = FALSE]
    /\ pc' = [pc EXCEPT ![i] = "ncs"]
    /\ UNCHANGED x

Idle == UNCHANGED vars

Next == \/ \E i \in Procs :
            Start(i) \/ SetX(i) \/ CheckY(i) \/ SetY(i)
         \/ CheckX(i) \/ WaitB(i) \/ Leave(i)
        \/ Idle

Spec == Init /\ [][Next]_vars

TypeOK ==
    /\ pc \in [Procs -> {"ncs","set_x","check_y","set_y","check_x","wait_b","cs"}]
    /\ b  \in [Procs -> BOOLEAN]
    /\ x  \in Procs \cup {NoProc}
    /\ y  \in Procs \cup {NoProc}
    /\ \A i, j \in Procs : (i # j /\ pc[i] = "cs") => pc[j] # "cs"
====
