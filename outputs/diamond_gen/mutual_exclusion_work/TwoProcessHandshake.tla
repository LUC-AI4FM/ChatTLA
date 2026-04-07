---- MODULE TwoProcessHandshake ----
(***************************************************************************)
(* A simple two-process handshake mutex.  Each process raises its request, *)
(* then enters the critical section only after observing that the other    *)
(* process is not currently requesting.                                    *)
(***************************************************************************)
EXTENDS Naturals

VARIABLES pc, req_a, req_b

vars == << pc, req_a, req_b >>

Init == /\ pc    = [p \in {"a","b"} |-> "ncs"]
        /\ req_a = FALSE
        /\ req_b = FALSE

\* A raises its request flag.
RaiseA ==
    /\ pc["a"] = "ncs"
    /\ req_a' = TRUE
    /\ pc' = [pc EXCEPT !["a"] = "wait"]
    /\ UNCHANGED req_b

\* A enters the CS only when B has not raised its flag.
EnterA ==
    /\ pc["a"] = "wait"
    /\ req_b = FALSE
    /\ pc' = [pc EXCEPT !["a"] = "cs"]
    /\ UNCHANGED << req_a, req_b >>

\* A backs off if it sees B requesting (gives priority to B).
BackoffA ==
    /\ pc["a"] = "wait"
    /\ req_b = TRUE
    /\ req_a' = FALSE
    /\ pc' = [pc EXCEPT !["a"] = "ncs"]
    /\ UNCHANGED req_b

LeaveA ==
    /\ pc["a"] = "cs"
    /\ req_a' = FALSE
    /\ pc' = [pc EXCEPT !["a"] = "ncs"]
    /\ UNCHANGED req_b

\* Symmetric for B, but we break ties by giving B priority over A on conflict.
RaiseB ==
    /\ pc["b"] = "ncs"
    /\ req_b' = TRUE
    /\ pc' = [pc EXCEPT !["b"] = "wait"]
    /\ UNCHANGED req_a

EnterB ==
    /\ pc["b"] = "wait"
    /\ pc["a"] # "cs"  \* extra check: don't enter if A is in cs
    /\ pc' = [pc EXCEPT !["b"] = "cs"]
    /\ UNCHANGED << req_a, req_b >>

LeaveB ==
    /\ pc["b"] = "cs"
    /\ req_b' = FALSE
    /\ pc' = [pc EXCEPT !["b"] = "ncs"]
    /\ UNCHANGED req_a

Next == RaiseA \/ EnterA \/ BackoffA \/ LeaveA
     \/ RaiseB \/ EnterB \/ LeaveB

Spec == Init /\ [][Next]_vars

TypeOK ==
    /\ pc \in [{"a","b"} -> {"ncs","wait","cs"}]
    /\ req_a \in BOOLEAN
    /\ req_b \in BOOLEAN
    /\ \A p, q \in {"a","b"} : (p # q /\ pc[p] = "cs") => pc[q] # "cs"
====
