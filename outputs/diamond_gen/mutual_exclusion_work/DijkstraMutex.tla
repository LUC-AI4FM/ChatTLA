---- MODULE DijkstraMutex ----
(***************************************************************************)
(* Dijkstra's original 1965 mutex for N processes.  Each process owns      *)
(* b[i] (busy) and c[i] (claim).  A shared variable k indicates which      *)
(* process is currently favoured.  This is the first software solution     *)
(* for N>2 processes.                                                      *)
(***************************************************************************)
EXTENDS Naturals

N == 2
Procs == 1..N

VARIABLES pc, b, c, k

vars == << pc, b, c, k >>

Init == /\ pc = [i \in Procs |-> "ncs"]
        /\ b  = [i \in Procs |-> TRUE]
        /\ c  = [i \in Procs |-> TRUE]
        /\ k  \in Procs

\* L1: announce intent by clearing b[i].
Start(i) ==
    /\ pc[i] = "ncs"
    /\ b' = [b EXCEPT ![i] = FALSE]
    /\ pc' = [pc EXCEPT ![i] = "L2"]
    /\ UNCHANGED << c, k >>

\* L2: if k # i, retreat (set c[i]=TRUE) and try to set k=i if b[k]=TRUE.
L2_retreat(i) ==
    /\ pc[i] = "L2"
    /\ k # i
    /\ c' = [c EXCEPT ![i] = TRUE]
    /\ IF b[k] = TRUE
         THEN /\ k' = i
              /\ pc' = [pc EXCEPT ![i] = "L2"]
         ELSE /\ k' = k
              /\ pc' = [pc EXCEPT ![i] = "L2"]
    /\ UNCHANGED b

\* L2: if k = i, advance to L3.
L2_advance(i) ==
    /\ pc[i] = "L2"
    /\ k = i
    /\ pc' = [pc EXCEPT ![i] = "L3"]
    /\ UNCHANGED << b, c, k >>

\* L3: claim by clearing c[i].
L3(i) ==
    /\ pc[i] = "L3"
    /\ c' = [c EXCEPT ![i] = FALSE]
    /\ pc' = [pc EXCEPT ![i] = "L4"]
    /\ UNCHANGED << b, k >>

\* L4: enter critical section iff no other process has c[j] = FALSE.
L4(i) ==
    /\ pc[i] = "L4"
    /\ \A j \in Procs \ {i} : c[j] = TRUE
    /\ pc' = [pc EXCEPT ![i] = "cs"]
    /\ UNCHANGED << b, c, k >>

\* L4 fail: another j has c[j]=FALSE — go back to L2.
L4_fail(i) ==
    /\ pc[i] = "L4"
    /\ \E j \in Procs \ {i} : c[j] = FALSE
    /\ c' = [c EXCEPT ![i] = TRUE]
    /\ b' = [b EXCEPT ![i] = TRUE]
    /\ pc' = [pc EXCEPT ![i] = "L2"]
    /\ UNCHANGED k

\* Leave: reset b and c for this process.
Leave(i) ==
    /\ pc[i] = "cs"
    /\ b' = [b EXCEPT ![i] = TRUE]
    /\ c' = [c EXCEPT ![i] = TRUE]
    /\ pc' = [pc EXCEPT ![i] = "ncs"]
    /\ UNCHANGED k

Next == \E i \in Procs :
           Start(i) \/ L2_retreat(i) \/ L2_advance(i) \/ L3(i)
        \/ L4(i) \/ L4_fail(i) \/ Leave(i)

Spec == Init /\ [][Next]_vars

TypeOK ==
    /\ pc \in [Procs -> {"ncs","L2","L3","L4","cs"}]
    /\ b  \in [Procs -> BOOLEAN]
    /\ c  \in [Procs -> BOOLEAN]
    /\ k  \in Procs
    /\ \A i, j \in Procs : (i # j /\ pc[i] = "cs") => pc[j] # "cs"
====
