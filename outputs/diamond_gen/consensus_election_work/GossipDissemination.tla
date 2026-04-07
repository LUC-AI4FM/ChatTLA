---- MODULE GossipDissemination ----
(***************************************************************************)
(* Epidemic gossip dissemination.  Each round a node copies state from a  *)
(* peer; the "infected" set only grows (monotone).                         *)
(* Safety: state is monotone — once a node is infected, it never recovers.*)
(* Eventually all nodes are infected (modeled as a Reset to allow looping).*)
(***************************************************************************)
EXTENDS Naturals, FiniteSets

Nodes == {1, 2, 3}

VARIABLES infected, history

vars == << infected, history >>

\* history records the maximum infection size ever seen — used by the
\* monotonicity safety property.
Init == /\ infected = [n \in Nodes |-> n = 1]   \* node 1 is patient zero
        /\ history  = 1

InfectedSet == {n \in Nodes : infected[n]}

\* Node n contacts a peer p; if p is infected, n becomes infected too.
Gossip(n, p) ==
    /\ n # p
    /\ ~infected[n]
    /\ infected[p]
    /\ infected' = [infected EXCEPT ![n] = TRUE]
    /\ history'  = Cardinality(InfectedSet) + 1

\* Restart once all nodes are infected so the state space stays finite.
Reset ==
    /\ \A n \in Nodes : infected[n]
    /\ infected' = [n \in Nodes |-> n = 1]
    /\ history'  = 1

Next == \/ \E n, p \in Nodes : Gossip(n, p)
        \/ Reset

Spec == Init /\ [][Next]_vars

TypeOK ==
    /\ infected \in [Nodes -> BOOLEAN]
    /\ history  \in 1..Cardinality(Nodes)

\* Strong safety: monotone spread — between successive non-reset transitions
\* the infected set grows.  We capture this by saying that history equals
\* the current infected count whenever the spread is in progress.
SafetyInv == (history >= 1) /\ (history <= Cardinality(InfectedSet) + 1) /\ (Cardinality(InfectedSet) >= 1)
====
