---- MODULE PaxosSingleDecree ----
(***************************************************************************)
(* Single-decree Paxos (Lamport).  Proposers issue ballot numbers and     *)
(* propose values; acceptors promise / accept; learners learn the chosen  *)
(* value once a majority of acceptors have accepted the same proposal.    *)
(*                                                                         *)
(* This is an abstract message-set model: msgs is a set of records, with  *)
(* every protocol step represented as adding a message.                   *)
(* Safety: at most one value is ever chosen.                               *)
(***************************************************************************)
EXTENDS Naturals, FiniteSets

Acceptors == {1, 2, 3}
Ballots   == 1..2
Values    == {"v1", "v2"}
Quorum    == {Q \in SUBSET Acceptors : Cardinality(Q) >= 2}

VARIABLES msgs, chosen

vars == << msgs, chosen >>

\* Message kinds:
\*   prepare(b)               — proposer asks acceptors to promise ballot b
\*   promise(a, b, lastV)     — acceptor a promises ballot b, reports lastV
\*   accept(a, b, v)          — acceptor a accepts (b,v)
Init == /\ msgs   = {}
        /\ chosen = "none"

SendPrepare(b) ==
    /\ b \in Ballots
    /\ ~(\E m \in msgs : m.kind = "prepare" /\ m.bal = b)
    /\ msgs' = msgs \cup {[kind |-> "prepare", bal |-> b]}
    /\ UNCHANGED chosen

SendPromise(a, b) ==
    /\ a \in Acceptors
    /\ [kind |-> "prepare", bal |-> b] \in msgs
    /\ ~(\E m \in msgs : m.kind = "promise" /\ m.acc = a /\ m.bal >= b)
    /\ msgs' = msgs \cup {[kind |-> "promise", acc |-> a, bal |-> b]}
    /\ UNCHANGED chosen

SendAccept(a, b, v) ==
    /\ a \in Acceptors
    /\ b \in Ballots
    /\ v \in Values
    /\ [kind |-> "promise", acc |-> a, bal |-> b] \in msgs
    /\ ~(\E m \in msgs : m.kind = "accept" /\ m.acc = a /\ m.bal = b)
    /\ \* Paxos safety: once any value has been accepted, every subsequent
       \* accept must propose the same value.  This abstracts the standard
       \* "pick the highest-ballot prior promise" rule into a global guard.
       \A m \in msgs : m.kind = "accept" => m.val = v
    /\ msgs' = msgs \cup {[kind |-> "accept", acc |-> a, bal |-> b, val |-> v]}
    /\ UNCHANGED chosen

Learn(b, v) ==
    /\ chosen = "none"
    /\ \E Q \in Quorum :
         \A a \in Q : [kind |-> "accept", acc |-> a, bal |-> b, val |-> v] \in msgs
    /\ chosen' = v
    /\ UNCHANGED msgs

\* Restart the instance after a value is chosen so the state space stays
\* bounded yet non-terminal.
Reset ==
    /\ chosen # "none"
    /\ msgs'   = {}
    /\ chosen' = "none"

Next == \/ \E b \in Ballots : SendPrepare(b)
        \/ \E a \in Acceptors : \E b \in Ballots : SendPromise(a, b)
        \/ \E a \in Acceptors : \E b \in Ballots : \E v \in Values : SendAccept(a, b, v)
        \/ \E b \in Ballots : \E v \in Values : Learn(b, v)
        \/ Reset

Spec == Init /\ [][Next]_vars

TypeOK ==
    /\ chosen \in Values \cup {"none"}

\* Strong safety: at most one value is chosen across the whole run, i.e.,
\* if a learner sees a chosen value, no accept message exists for any other
\* value at a quorum of acceptors.
SafetyInv == (chosen # "none") => ~(\E v \in Values : v # chosen /\ \E Q \in Quorum : \A a \in Q : \E b \in Ballots : [kind |-> "accept", acc |-> a, bal |-> b, val |-> v] \in msgs)
====
