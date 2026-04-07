---- MODULE DekkerMutex ----
(***************************************************************************)
(* Dekker's algorithm (1965) — the first software-only mutex for two       *)
(* processes.  Each process raises its flag, then if it sees the other's   *)
(* flag it consults `turn` to decide whether to back off and spin.         *)
(***************************************************************************)
EXTENDS Naturals

Procs == {0, 1}

VARIABLES pc, flag, turn

vars == << pc, flag, turn >>

Other(i) == 1 - i

Init == /\ pc   = [i \in Procs |-> "ncs"]
        /\ flag = [i \in Procs |-> FALSE]
        /\ turn = 0

\* Raise our flag, indicating intent to enter.
Raise(i) ==
    /\ pc[i] = "ncs"
    /\ flag' = [flag EXCEPT ![i] = TRUE]
    /\ pc'   = [pc   EXCEPT ![i] = "check"]
    /\ UNCHANGED turn

\* Check the other's flag.  If clear, enter; otherwise consult turn.
CheckOther(i) ==
    /\ pc[i] = "check"
    /\ IF flag[Other(i)] = FALSE
         THEN pc' = [pc EXCEPT ![i] = "cs"]
         ELSE pc' = [pc EXCEPT ![i] = "consult"]
    /\ UNCHANGED << flag, turn >>

\* If it isn't our turn, lower our flag and spin until turn becomes i.
Yield(i) ==
    /\ pc[i] = "consult"
    /\ turn # i
    /\ flag' = [flag EXCEPT ![i] = FALSE]
    /\ pc'   = [pc   EXCEPT ![i] = "wait_turn"]
    /\ UNCHANGED turn

\* Wait until our turn, then re-raise our flag and re-check.
TakeTurn(i) ==
    /\ pc[i] = "wait_turn"
    /\ turn = i
    /\ flag' = [flag EXCEPT ![i] = TRUE]
    /\ pc'   = [pc   EXCEPT ![i] = "check"]
    /\ UNCHANGED turn

\* If it IS our turn while in consult, just keep our flag up and proceed.
Persist(i) ==
    /\ pc[i] = "consult"
    /\ turn = i
    /\ pc' = [pc EXCEPT ![i] = "check"]
    /\ UNCHANGED << flag, turn >>

\* Leave: pass turn to the other process and lower our flag.
Leave(i) ==
    /\ pc[i] = "cs"
    /\ pc'   = [pc   EXCEPT ![i] = "ncs"]
    /\ flag' = [flag EXCEPT ![i] = FALSE]
    /\ turn' = Other(i)

Next == \E i \in Procs :
           Raise(i) \/ CheckOther(i) \/ Yield(i)
        \/ TakeTurn(i) \/ Persist(i) \/ Leave(i)

Spec == Init /\ [][Next]_vars

TypeOK ==
    /\ pc   \in [Procs -> {"ncs","check","consult","wait_turn","cs"}]
    /\ flag \in [Procs -> BOOLEAN]
    /\ turn \in Procs
    /\ \A i, j \in Procs : (i # j /\ pc[i] = "cs") => pc[j] # "cs"
====
