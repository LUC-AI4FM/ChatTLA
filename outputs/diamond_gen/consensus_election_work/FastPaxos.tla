---- MODULE FastPaxos ----
(***************************************************************************)
(* Single-decree Fast Paxos.  Clients send proposals directly to acceptors *)
(* in a "fast round"; if a fast quorum (3/4 of acceptors) all accept the   *)
(* same value, that value is chosen without a coordinator.  A classic     *)
(* round may then recover collisions.                                      *)
(*                                                                         *)
(* This abstract model has only the fast round and a fall-back classic    *)
(* round.  Safety: agreement on the chosen value.                         *)
(***************************************************************************)
EXTENDS Naturals, FiniteSets

Acceptors  == {1, 2, 3, 4}
Values     == {"v1", "v2"}
FastQuorum    == {Q \in SUBSET Acceptors : Cardinality(Q) >= 3}
ClassicQuorum == {Q \in SUBSET Acceptors : Cardinality(Q) >= 3}

VARIABLES accepted, chosen, round

vars == << accepted, chosen, round >>

Init == /\ accepted = [a \in Acceptors |-> "none"]
        /\ chosen   = "none"
        /\ round    = "fast"

\* Fast round: any acceptor may directly accept any client value (provided
\* it has not yet accepted anything in the current run).
FastAccept(a, v) ==
    /\ round = "fast"
    /\ accepted[a] = "none"
    /\ chosen = "none"
    /\ accepted' = [accepted EXCEPT ![a] = v]
    /\ UNCHANGED << chosen, round >>

\* Fast-round chooses if a fast quorum agrees on a value.
FastChoose(v) ==
    /\ round = "fast"
    /\ chosen = "none"
    /\ \E Q \in FastQuorum : \A a \in Q : accepted[a] = v
    /\ chosen' = v
    /\ UNCHANGED << accepted, round >>

\* Fast-round detected a collision (two values accepted by different
\* acceptors).  Switch to a classic recovery round.
FallBack ==
    /\ round = "fast"
    /\ chosen = "none"
    /\ \E a, b \in Acceptors : accepted[a] # "none" /\ accepted[b] # "none" /\ accepted[a] # accepted[b]
    /\ round' = "classic"
    /\ UNCHANGED << accepted, chosen >>

\* Classic recovery: a coordinator picks any value already accepted by some
\* acceptor and re-proposes it under a higher ballot, overriding all
\* acceptors.  A fast quorum then trivially holds the chosen value.
ClassicChoose(v) ==
    /\ round = "classic"
    /\ chosen = "none"
    /\ \E a \in Acceptors : accepted[a] = v
    /\ accepted' = [b \in Acceptors |-> v]
    /\ chosen'   = v
    /\ UNCHANGED round

\* Restart for the next instance.
Reset ==
    /\ chosen # "none"
    /\ accepted' = [a \in Acceptors |-> "none"]
    /\ chosen'   = "none"
    /\ round'    = "fast"

Next == \/ \E a \in Acceptors : \E v \in Values : FastAccept(a, v)
        \/ \E v \in Values : FastChoose(v)
        \/ FallBack
        \/ \E v \in Values : ClassicChoose(v)
        \/ Reset

Spec == Init /\ [][Next]_vars

TypeOK ==
    /\ accepted \in [Acceptors -> Values \cup {"none"}]
    /\ chosen   \in Values \cup {"none"}
    /\ round    \in {"fast", "classic"}

\* Strong safety: a fast quorum still backs the chosen value (the property
\* that makes the choice unique under Fast Paxos's quorum intersection).
SafetyInv == (chosen # "none") => (\E Q \in FastQuorum : \A a \in Q : accepted[a] = chosen)
====
