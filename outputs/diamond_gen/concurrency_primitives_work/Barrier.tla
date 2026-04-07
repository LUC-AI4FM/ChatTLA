---- MODULE Barrier ----
EXTENDS Naturals, FiniteSets

CONSTANT Procs

\* arrived = set of processes currently waiting at the barrier
\* past    = set of processes that have crossed the barrier this round
VARIABLES arrived, past

vars == << arrived, past >>

N == Cardinality(Procs)

Init == /\ arrived = {}
        /\ past    = {}

\* A process not yet at the barrier arrives.
Arrive(p) == /\ p \notin arrived
             /\ p \notin past
             /\ arrived' = arrived \cup {p}
             /\ UNCHANGED past

\* When all N processes have arrived, the barrier opens: they all become past.
Release == /\ Cardinality(arrived) = N
           /\ past' = arrived
           /\ arrived' = {}

\* New round: once all are past, reset for the next iteration.
Reset == /\ Cardinality(past) = N
         /\ past' = {}
         /\ UNCHANGED arrived

Next == \/ \E p \in Procs : Arrive(p)
        \/ Release
        \/ Reset

Spec == Init /\ [][Next]_vars

\* Safety: no process is past the barrier unless all N have arrived
\* this round. Equivalently, past is either empty or has exactly N members.
BarrierSafe == /\ (past = {}) \/ (Cardinality(past) = N)
               /\ (arrived \cap past) = {}

TypeOK == /\ arrived \subseteq Procs
          /\ past    \subseteq Procs
          /\ BarrierSafe
====
