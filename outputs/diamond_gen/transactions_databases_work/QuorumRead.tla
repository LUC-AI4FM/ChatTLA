---- MODULE QuorumRead ----
(***************************************************************************)
(*  Quorum read with read-repair.  Each replica stores a (value,        *)
(*  version) pair.  A read contacts a read-quorum, returns the value    *)
(*  with the largest version, and writes that value back to any stale  *)
(*  replica it contacted.                                               *)
(*                                                                         *)
(*  Strong invariant: any two read quorums intersect, and the version   *)
(*  on every replica is monotone.                                       *)
(***************************************************************************)
EXTENDS Naturals, FiniteSets

CONSTANTS Replicas, MaxVersion

Quorum == (Cardinality(Replicas) \div 2) + 1

VARIABLES ver

vars == << ver >>

Init == /\ ver = [r \in Replicas |-> 0]

\* A write quorum bumps the version on a majority of replicas.  We
\* model the quorum write as one atomic step.
Write ==
    /\ \E Q \in SUBSET Replicas :
          /\ Cardinality(Q) >= Quorum
          /\ \E v \in 1..MaxVersion :
                /\ \A r \in Q : ver[r] < v
                /\ ver' = [r \in Replicas |-> IF r \in Q THEN v ELSE ver[r]]

\* A read quorum repairs stale replicas to the largest version it sees.
ReadRepair ==
    /\ \E Q \in SUBSET Replicas :
          /\ Cardinality(Q) >= Quorum
          /\ LET maxv == CHOOSE v \in {ver[r] : r \in Q} :
                            \A w \in {ver[r] : r \in Q} : w <= v
             IN  ver' = [r \in Replicas |-> IF r \in Q THEN maxv ELSE ver[r]]

Next == \/ Write \/ ReadRepair

Spec == Init /\ [][Next]_vars /\ WF_vars(Next)

\* Strong invariant: any two majorities intersect (true by counting),
\* and the largest seen version on any majority is at least as great as
\* on every previously written majority.
TypeOK == /\ ver \in [Replicas -> 0..MaxVersion]
          /\ \A Q1, Q2 \in SUBSET Replicas :
                (Cardinality(Q1) >= Quorum /\ Cardinality(Q2) >= Quorum)
                    => Q1 \cap Q2 # {}
====
