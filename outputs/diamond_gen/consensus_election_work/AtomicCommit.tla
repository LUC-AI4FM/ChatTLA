---- MODULE AtomicCommit ----
(***************************************************************************)
(* Generic atomic commit gate (the "AC" abstract problem).  All           *)
(* participants must reach the same decision: commit if everyone votes    *)
(* yes, otherwise abort.  No coordinator is modeled — this is the         *)
(* abstract specification of uniform agreement that any 2PC, 3PC, or     *)
(* Paxos-Commit refines.                                                 *)
(* Safety: uniform agreement on the decision.                             *)
(***************************************************************************)
EXTENDS Naturals, FiniteSets

Participants == {1, 2, 3}
Decisions    == {"commit", "abort"}

VARIABLES vote, decision

vars == << vote, decision >>

Init == /\ vote     = [p \in Participants |-> "none"]
        /\ decision = [p \in Participants |-> "none"]

CastVote(p, v) ==
    /\ vote[p] = "none"
    /\ v \in {"yes", "no"}
    /\ vote' = [vote EXCEPT ![p] = v]
    /\ UNCHANGED decision

\* Decide commit only if every participant has voted yes.
DecideCommit(p) ==
    /\ decision[p] = "none"
    /\ \A q \in Participants : vote[q] = "yes"
    /\ decision' = [decision EXCEPT ![p] = "commit"]
    /\ UNCHANGED vote

\* Decide abort if some participant has voted no.
DecideAbort(p) ==
    /\ decision[p] = "none"
    /\ \E q \in Participants : vote[q] = "no"
    /\ \A q \in Participants : vote[q] # "none"
    /\ decision' = [decision EXCEPT ![p] = "abort"]
    /\ UNCHANGED vote

\* Restart so the state space stays finite without terminal states.
Reset ==
    /\ \A p \in Participants : decision[p] \in Decisions
    /\ vote'     = [p \in Participants |-> "none"]
    /\ decision' = [p \in Participants |-> "none"]

Next == \/ \E p \in Participants : \E v \in {"yes", "no"} : CastVote(p, v)
        \/ \E p \in Participants : DecideCommit(p)
        \/ \E p \in Participants : DecideAbort(p)
        \/ Reset

Spec == Init /\ [][Next]_vars

TypeOK ==
    /\ vote     \in [Participants -> {"none", "yes", "no"}]
    /\ decision \in [Participants -> {"none", "commit", "abort"}]

\* Strong safety: uniform agreement among participants that have decided.
SafetyInv == \A p, q \in Participants : (decision[p] \in Decisions /\ decision[q] \in Decisions) => decision[p] = decision[q]
====
