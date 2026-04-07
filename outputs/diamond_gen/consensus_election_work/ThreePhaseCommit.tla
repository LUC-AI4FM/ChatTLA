---- MODULE ThreePhaseCommit ----
(***************************************************************************)
(* Three-phase commit (Skeen, 1981) — adds a "pre-commit" stage between   *)
(* the vote phase and the final commit so a crashed coordinator never     *)
(* leaves participants blocked indefinitely.                              *)
(*                                                                         *)
(* Phases per participant: working -> voted -> precommit -> commit/abort. *)
(* Coordinator phases:     init    -> ready -> precommit -> done.         *)
(* Safety: no two participants ever decide differently.                   *)
(***************************************************************************)
EXTENDS Naturals, FiniteSets

Participants == {1, 2, 3}
Decisions    == {"commit", "abort"}

VARIABLES pState, pVote, cPhase, cDecision

vars == << pState, pVote, cPhase, cDecision >>

Init == /\ pState    = [p \in Participants |-> "working"]
        /\ pVote     = [p \in Participants |-> "none"]
        /\ cPhase    = "init"
        /\ cDecision = "none"

Vote(p, v) ==
    /\ pState[p] = "working"
    /\ v \in {"yes", "no"}
    /\ pVote'  = [pVote  EXCEPT ![p] = v]
    /\ pState' = [pState EXCEPT ![p] = "voted"]
    /\ UNCHANGED << cPhase, cDecision >>

\* Coordinator moves to ready once all votes are in.
Collect ==
    /\ cPhase = "init"
    /\ \A p \in Participants : pVote[p] # "none"
    /\ cPhase' = "ready"
    /\ UNCHANGED << pState, pVote, cDecision >>

\* Coordinator broadcasts pre-commit when every vote is YES.
PreCommit ==
    /\ cPhase = "ready"
    /\ \A p \in Participants : pVote[p] = "yes"
    /\ cPhase' = "precommit"
    /\ UNCHANGED << pState, pVote, cDecision >>

\* Coordinator broadcasts abort when at least one vote is NO.
AbortAll ==
    /\ cPhase = "ready"
    /\ \E p \in Participants : pVote[p] = "no"
    /\ cPhase'    = "done"
    /\ cDecision' = "abort"
    /\ UNCHANGED << pState, pVote >>

\* A participant acks the pre-commit and moves into the precommit phase.
EnterPreCommit(p) ==
    /\ cPhase = "precommit"
    /\ pState[p] = "voted"
    /\ pState' = [pState EXCEPT ![p] = "precommit"]
    /\ UNCHANGED << pVote, cPhase, cDecision >>

\* Coordinator finalises commit after all participants have pre-committed.
DoCommit ==
    /\ cPhase = "precommit"
    /\ \A p \in Participants : pState[p] = "precommit"
    /\ cPhase'    = "done"
    /\ cDecision' = "commit"
    /\ UNCHANGED << pState, pVote >>

\* Each participant adopts the final decision.
Decide(p) ==
    /\ cPhase = "done"
    /\ pState[p] \notin Decisions
    /\ pState' = [pState EXCEPT ![p] = cDecision]
    /\ UNCHANGED << pVote, cPhase, cDecision >>

\* Restart for the next transaction (avoids terminal states).
Reset ==
    /\ cPhase = "done"
    /\ \A p \in Participants : pState[p] \in Decisions
    /\ pState'    = [p \in Participants |-> "working"]
    /\ pVote'     = [p \in Participants |-> "none"]
    /\ cPhase'    = "init"
    /\ cDecision' = "none"

Next == \/ \E p \in Participants : \E v \in {"yes", "no"} : Vote(p, v)
        \/ Collect
        \/ PreCommit
        \/ AbortAll
        \/ \E p \in Participants : EnterPreCommit(p)
        \/ DoCommit
        \/ \E p \in Participants : Decide(p)
        \/ Reset

Spec == Init /\ [][Next]_vars

TypeOK ==
    /\ pState    \in [Participants -> {"working", "voted", "precommit", "commit", "abort"}]
    /\ pVote     \in [Participants -> {"none", "yes", "no"}]
    /\ cPhase    \in {"init", "ready", "precommit", "done"}
    /\ cDecision \in {"none", "commit", "abort"}

\* Strong safety: uniform agreement on the decision and consistency with
\* the coordinator's choice.
SafetyInv == (\A p, q \in Participants : (pState[p] \in Decisions /\ pState[q] \in Decisions) => pState[p] = pState[q]) /\ (\A p \in Participants : pState[p] \in Decisions => pState[p] = cDecision)
====
