---- MODULE BurnsMutex ----
(***************************************************************************)
(* Burns' mutual exclusion algorithm (1981) for N processes using only a   *)
(* single binary flag per process.  Lower-id processes have priority: a    *)
(* process at level "wait" enters the CS only when no lower-id flag is set *)
(* AND no other process is already in the CS.                              *)
(***************************************************************************)
EXTENDS Naturals

N == 3
Procs == 1..N

VARIABLES pc, flag

vars == << pc, flag >>

Init == /\ pc   = [i \in Procs |-> "ncs"]
        /\ flag = [i \in Procs |-> FALSE]

\* Step 1: clear our flag and check for higher-priority (lower id) competitors.
ClearAndCheck(i) ==
    /\ pc[i] = "ncs"
    /\ flag' = [flag EXCEPT ![i] = FALSE]
    /\ pc'   = [pc EXCEPT ![i] = "check_low"]

\* Step 2: if any lower-id process has its flag up, restart.
ScanLow(i) ==
    /\ pc[i] = "check_low"
    /\ IF \E j \in Procs : j < i /\ flag[j]
         THEN pc' = [pc EXCEPT ![i] = "ncs"]
         ELSE pc' = [pc EXCEPT ![i] = "set_flag"]
    /\ UNCHANGED flag

\* Step 3: raise our flag.
SetFlag(i) ==
    /\ pc[i] = "set_flag"
    /\ flag' = [flag EXCEPT ![i] = TRUE]
    /\ pc'   = [pc EXCEPT ![i] = "recheck_low"]

\* Step 4: re-check lower-id flags after we set ours.
RecheckLow(i) ==
    /\ pc[i] = "recheck_low"
    /\ IF \E j \in Procs : j < i /\ flag[j]
         THEN /\ flag' = [flag EXCEPT ![i] = FALSE]
              /\ pc'   = [pc EXCEPT ![i] = "ncs"]
         ELSE /\ pc'   = [pc EXCEPT ![i] = "wait_high"]
              /\ UNCHANGED flag

\* Step 5: wait until no higher-id process has its flag up.
WaitHigh(i) ==
    /\ pc[i] = "wait_high"
    /\ \A j \in Procs : j > i => ~flag[j]
    /\ pc' = [pc EXCEPT ![i] = "cs"]
    /\ UNCHANGED flag

Leave(i) ==
    /\ pc[i] = "cs"
    /\ flag' = [flag EXCEPT ![i] = FALSE]
    /\ pc'   = [pc EXCEPT ![i] = "ncs"]

Next == \E i \in Procs :
            ClearAndCheck(i) \/ ScanLow(i) \/ SetFlag(i)
         \/ RecheckLow(i) \/ WaitHigh(i) \/ Leave(i)

Spec == Init /\ [][Next]_vars

TypeOK ==
    /\ pc   \in [Procs -> {"ncs","check_low","set_flag","recheck_low","wait_high","cs"}]
    /\ flag \in [Procs -> BOOLEAN]
    /\ \A i, j \in Procs : (i # j /\ pc[i] = "cs") => pc[j] # "cs"
====
