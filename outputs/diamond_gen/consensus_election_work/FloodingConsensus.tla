---- MODULE FloodingConsensus ----
(***************************************************************************)
(* Synchronous flooding consensus tolerating f crash failures.            *)
(* Each round every alive node broadcasts its current set of known        *)
(* values; after f+1 rounds, all alive nodes hold the same set, hence the *)
(* same minimum element.  We model the rounds explicitly with a global    *)
(* round counter and per-node "known" sets.                               *)
(* Safety: any two alive nodes that decide pick the same value (validity  *)
(* + agreement).                                                          *)
(***************************************************************************)
EXTENDS Naturals, FiniteSets

Nodes == {1, 2, 3}
Values == {1, 2, 3}
F == 1
MaxRound == F + 1     \* f+1 = 2 rounds suffice

VARIABLES known, alive, round, decision

vars == << known, alive, round, decision >>

Init == /\ known    = [n \in Nodes |-> {n}]    \* node n initially proposes its id
        /\ alive    = [n \in Nodes |-> TRUE]
        /\ round    = 0
        /\ decision = [n \in Nodes |-> 0]

Min(S) == CHOOSE x \in S : \A y \in S : x <= y

\* One synchronous round: every alive node merges in every other alive
\* node's known set, simulating perfect broadcast.  Round counter advances.
Flood ==
    /\ round < MaxRound
    /\ \A n \in Nodes : decision[n] = 0
    /\ known' = [k \in Nodes |-> IF alive[k]
                                  THEN UNION { known[m] : m \in {x \in Nodes : alive[x]} }
                                  ELSE known[k]]
    /\ round' = round + 1
    /\ UNCHANGED << alive, decision >>

\* After f+1 rounds, alive nodes commit to the minimum of their known set.
Decide(n) ==
    /\ round = MaxRound
    /\ alive[n]
    /\ decision[n] = 0
    /\ decision' = [decision EXCEPT ![n] = Min(known[n])]
    /\ UNCHANGED << known, alive, round >>

\* A node may crash at any time before deciding (bounded by F).
Crash(n) ==
    /\ alive[n]
    /\ Cardinality({m \in Nodes : ~alive[m]}) < F
    /\ alive' = [alive EXCEPT ![n] = FALSE]
    /\ UNCHANGED << known, round, decision >>

\* Restart for the next consensus instance.
Reset ==
    /\ \A n \in Nodes : (~alive[n]) \/ decision[n] # 0
    /\ round = MaxRound
    /\ known'    = [n \in Nodes |-> {n}]
    /\ alive'    = [n \in Nodes |-> TRUE]
    /\ round'    = 0
    /\ decision' = [n \in Nodes |-> 0]

Next == \/ Flood
        \/ \E d \in Nodes : Decide(d)
        \/ \E c \in Nodes : Crash(c)
        \/ Reset

Spec == Init /\ [][Next]_vars

TypeOK ==
    /\ known    \in [Nodes -> SUBSET Values]
    /\ alive    \in [Nodes -> BOOLEAN]
    /\ round    \in 0..MaxRound
    /\ decision \in [Nodes -> 0..3]

\* Strong safety: any two alive nodes that have decided agree on the value.
SafetyInv == \A m, n \in Nodes : (alive[m] /\ alive[n] /\ decision[m] # 0 /\ decision[n] # 0) => decision[m] = decision[n]
====
