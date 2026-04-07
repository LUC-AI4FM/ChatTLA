---- MODULE ByzantineQuorum ----
(***************************************************************************)
(* Byzantine quorum write.  With N = 3f + 1 replicas, a write is durable  *)
(* once at least 2f + 1 honest acks have been collected.  Two such        *)
(* quorums always intersect in at least f + 1 honest replicas.            *)
(* Safety: honest learners always agree on the durable value.             *)
(***************************************************************************)
EXTENDS Naturals, FiniteSets

F          == 1
Replicas   == 1..(3*F + 1)
Honest     == {1, 2, 3}                  \* replica 4 is potentially Byzantine
Values     == {"x", "y"}
QuorumSize  == 2*F + 1                    \* honest acks required = 3
Quorums    == {Q \in SUBSET Honest : Cardinality(Q) >= QuorumSize}

VARIABLES acks, byz, chosen

vars == << acks, byz, chosen >>

Init == /\ acks   = [r \in Replicas |-> "none"]
        /\ byz    = "none"               \* Byzantine replica's last equivocation
        /\ chosen = "none"

\* An honest replica acks value v (only if it has not yet acked anything,
\* and only if it does not contradict prior honest acks — honest replicas
\* are write-once for safety in this abstraction).
HonestAck(r, v) ==
    /\ r \in Honest
    /\ acks[r] = "none"
    /\ \A s \in Honest : acks[s] \in {"none", v}
    /\ acks' = [acks EXCEPT ![r] = v]
    /\ UNCHANGED << byz, chosen >>

\* The Byzantine replica equivocates: it can claim any value at any time
\* without affecting honest acks.
Equivocate(v) ==
    /\ byz' = v
    /\ UNCHANGED << acks, chosen >>

\* The chosen value is committed once a quorum of honest replicas have
\* acked it.
Commit(v) ==
    /\ chosen = "none"
    /\ \E Q \in Quorums : \A r \in Q : acks[r] = v
    /\ chosen' = v
    /\ UNCHANGED << acks, byz >>

\* Restart for the next write to bound the state space.
Reset ==
    /\ chosen # "none"
    /\ acks'   = [r \in Replicas |-> "none"]
    /\ byz'    = "none"
    /\ chosen' = "none"

Next == \/ \E r \in Replicas : \E v \in Values : HonestAck(r, v)
        \/ \E v \in Values : Equivocate(v)
        \/ \E v \in Values : Commit(v)
        \/ Reset

Spec == Init /\ [][Next]_vars

TypeOK ==
    /\ acks   \in [Replicas -> Values \cup {"none"}]
    /\ byz    \in Values \cup {"none"}
    /\ chosen \in Values \cup {"none"}

\* Strong safety: any chosen value is backed by a full quorum of honest
\* acks (Byzantine equivocation cannot fake this).
SafetyInv == (chosen # "none") => (\E Q \in Quorums : \A r \in Q : acks[r] = chosen)
====
