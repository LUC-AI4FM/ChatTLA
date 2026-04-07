---- MODULE TwoPhaseCommit ----
(***************************************************************************)
(* The classic Two-Phase Commit protocol with one coordinator and N        *)
(* participants.  The coordinator collects votes from all participants     *)
(* and decides COMMIT iff all votes are YES, otherwise ABORT.  Each        *)
(* participant then learns the decision.                                   *)
(* Safety: no two participants ever decide differently (uniform agreement).*)
(***************************************************************************)
EXTENDS Naturals, FiniteSets

Participants == {1, 2, 3}
Decisions    == {"commit", "abort"}

VARIABLES pState, pVote, cState, cDecision

vars == << pState, pVote, cState, cDecision >>

Init == /\ pState    = [p \in Participants |-> "working"]
        /\ pVote     = [p \in Participants |-> "none"]
        /\ cState    = "init"
        /\ cDecision = "none"

\* A participant casts its vote.
Vote(p, v) ==
    /\ pState[p] = "working"
    /\ v \in {"yes", "no"}
    /\ pVote'  = [pVote  EXCEPT ![p] = v]
    /\ pState' = [pState EXCEPT ![p] = "voted"]
    /\ UNCHANGED << cState, cDecision >>

\* Coordinator decides commit when every participant voted YES.
DecideCommit ==
    /\ cState = "init"
    /\ \A p \in Participants : pVote[p] = "yes"
    /\ cState'    = "decided"
    /\ cDecision' = "commit"
    /\ UNCHANGED << pState, pVote >>

\* Coordinator decides abort when at least one participant voted NO and all
\* have voted (or any has voted no).
DecideAbort ==
    /\ cState = "init"
    /\ \E p \in Participants : pVote[p] = "no"
    /\ cState'    = "decided"
    /\ cDecision' = "abort"
    /\ UNCHANGED << pState, pVote >>

\* Each participant learns the coordinator's decision.
Learn(p) ==
    /\ cState = "decided"
    /\ pState[p] = "voted"
    /\ pState' = [pState EXCEPT ![p] = cDecision]
    /\ UNCHANGED << pVote, cState, cDecision >>

\* Restart a new transaction once everyone has decided, to avoid terminal
\* states and keep the state space finite.
Reset ==
    /\ cState = "decided"
    /\ \A p \in Participants : pState[p] \in Decisions
    /\ pState'    = [p \in Participants |-> "working"]
    /\ pVote'     = [p \in Participants |-> "none"]
    /\ cState'    = "init"
    /\ cDecision' = "none"

Next == \/ \E p \in Participants : \E v \in {"yes", "no"} : Vote(p, v)
        \/ DecideCommit
        \/ DecideAbort
        \/ \E p \in Participants : Learn(p)
        \/ Reset

Spec == Init /\ [][Next]_vars

TypeOK ==
    /\ pState    \in [Participants -> {"working", "voted", "commit", "abort"}]
    /\ pVote     \in [Participants -> {"none", "yes", "no"}]
    /\ cState    \in {"init", "decided"}
    /\ cDecision \in {"none", "commit", "abort"}

\* Strong safety: uniform agreement — no two participants disagree, and any
\* participant decision matches the coordinator's decision.
SafetyInv == (\A p, q \in Participants : (pState[p] \in Decisions /\ pState[q] \in Decisions) => pState[p] = pState[q]) /\ (\A p \in Participants : pState[p] \in Decisions => pState[p] = cDecision)
====
