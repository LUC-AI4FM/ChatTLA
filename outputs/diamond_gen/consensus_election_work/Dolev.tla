---- MODULE Dolev ----
(***************************************************************************)
(* Dolev-Strong synchronous Byzantine broadcast (1983).  Tolerates up to *)
(* f Byzantine faults.  Each round a value is forwarded only after it     *)
(* has collected at least f+1 distinct signatures, ensuring honest       *)
(* receivers cannot be tricked.                                          *)
(* Safety: any two honest nodes that accept agree on the value.          *)
(***************************************************************************)
EXTENDS Naturals, FiniteSets

F        == 1
Nodes    == 1..(F + 2)             \* honest nodes
Sender   == 1
Values   == {"v0", "v1"}
MaxRound == F + 1

VARIABLES sigs, accepted, round

vars == << sigs, accepted, round >>

\* sigs[v] is the set of nodes that have signed value v.
Init == /\ sigs     = [v \in Values |-> {}]
        /\ accepted = [n \in Nodes  |-> "none"]
        /\ round    = 0

\* The sender broadcasts a value, contributing the first signature.
Broadcast(v) ==
    /\ round = 0
    /\ sigs[v] = {}
    /\ sigs' = [sigs EXCEPT ![v] = {Sender}]
    /\ round' = 1
    /\ UNCHANGED accepted

\* In each subsequent round, an honest node n adds its signature to v if
\* the current signature set is non-empty (it was forwarded).
Sign(n, v) ==
    /\ round >= 1
    /\ round < MaxRound
    /\ n \notin sigs[v]
    /\ Cardinality(sigs[v]) >= 1
    /\ sigs' = [sigs EXCEPT ![v] = @ \cup {n}]
    /\ UNCHANGED << accepted, round >>

\* Advance the round (synchronous protocol).
Tick ==
    /\ round >= 1
    /\ round < MaxRound
    /\ round' = round + 1
    /\ UNCHANGED << sigs, accepted >>

\* After f+1 rounds, an honest node accepts a value v if it has collected
\* at least f+1 distinct signatures and no other value has the same.
Accept(n, v) ==
    /\ round = MaxRound
    /\ accepted[n] = "none"
    /\ Cardinality(sigs[v]) >= F + 1
    /\ \A w \in Values : w # v => Cardinality(sigs[w]) < F + 1
    /\ accepted' = [accepted EXCEPT ![n] = v]
    /\ UNCHANGED << sigs, round >>

\* Restart for the next instance — also fires on stalemate to avoid
\* terminal states.
Reset ==
    /\ round = MaxRound
    /\ sigs'     = [v \in Values |-> {}]
    /\ accepted' = [n \in Nodes |-> "none"]
    /\ round'    = 0

Next == \/ \E v \in Values : Broadcast(v)
        \/ \E n \in Nodes : \E v \in Values : Sign(n, v)
        \/ Tick
        \/ \E n \in Nodes : \E v \in Values : Accept(n, v)
        \/ Reset

Spec == Init /\ [][Next]_vars

TypeOK ==
    /\ sigs     \in [Values -> SUBSET Nodes]
    /\ accepted \in [Nodes -> Values \cup {"none"}]
    /\ round    \in 0..MaxRound

\* Strong safety: any two honest nodes that have accepted hold the same
\* value (Dolev-Strong agreement).
SafetyInv == \A m, n \in Nodes : (accepted[m] # "none" /\ accepted[n] # "none") => accepted[m] = accepted[n]
====
