---- MODULE QuorumWrite ----
(***************************************************************************)
(* Quorum-based write replication.  A write succeeds when at least W out  *)
(* of N replicas acknowledge.  Two majority quorums always intersect, so  *)
(* any chosen value is unique.                                            *)
(* Safety: at most one value is chosen, and any chosen value is held by   *)
(* every replica in some majority quorum.                                 *)
(***************************************************************************)
EXTENDS Naturals, FiniteSets

Replicas == {1, 2, 3}
Values   == {"a", "b"}
W        == 2                            \* majority threshold (W > N/2)
Quorums  == {Q \in SUBSET Replicas : Cardinality(Q) >= W}

VARIABLES store, chosen

vars == << store, chosen >>

Init == /\ store  = [r \in Replicas |-> "none"]
        /\ chosen = "none"

\* A replica accepts a write of value v iff no value has yet been chosen.
\* (This serializes proposals — a stricter version than real life.)
Accept(r, v) ==
    /\ store[r] = "none"
    /\ chosen   = "none"
    /\ \A s \in Replicas : store[s] \in {"none", v}
    /\ store' = [store EXCEPT ![r] = v]
    /\ UNCHANGED chosen

\* The value v is chosen once a write quorum holds it.
Commit(v) ==
    /\ chosen = "none"
    /\ \E Q \in Quorums : \A r \in Q : store[r] = v
    /\ chosen' = v
    /\ UNCHANGED store

\* Restart for the next write.
Reset ==
    /\ chosen # "none"
    /\ store'  = [r \in Replicas |-> "none"]
    /\ chosen' = "none"

Next == \/ \E r \in Replicas : \E v \in Values : Accept(r, v)
        \/ \E v \in Values : Commit(v)
        \/ Reset

Spec == Init /\ [][Next]_vars

TypeOK ==
    /\ store  \in [Replicas -> Values \cup {"none"}]
    /\ chosen \in Values \cup {"none"}

\* Strong safety: any two majority quorums intersect, so a chosen value is
\* visible to every other quorum that exists.
SafetyInv == (chosen # "none") => (\A Q \in Quorums : \E r \in Q : store[r] = chosen)
====
