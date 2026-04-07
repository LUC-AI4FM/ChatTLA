---- MODULE CoordinatorRecovery ----
(***************************************************************************)
(* Two-phase commit with coordinator recovery.  A primary coordinator may *)
(* crash mid-protocol and is replaced by a backup that resumes from the   *)
(* current decision (commit or abort).                                   *)
(* Safety: no participant ever sees a decision conflicting with another  *)
(* participant's decision (uniform agreement).                           *)
(***************************************************************************)
EXTENDS Naturals, FiniteSets

Participants == {1, 2, 3}
Decisions    == {"commit", "abort"}

VARIABLES pState, pVote, primary, backup, decision

vars == << pState, pVote, primary, backup, decision >>

Init == /\ pState   = [p \in Participants |-> "working"]
        /\ pVote    = [p \in Participants |-> "none"]
        /\ primary  = "alive"
        /\ backup   = "standby"
        /\ decision = "none"

Vote(p, v) ==
    /\ pState[p] = "working"
    /\ v \in {"yes", "no"}
    /\ pVote'  = [pVote  EXCEPT ![p] = v]
    /\ pState' = [pState EXCEPT ![p] = "voted"]
    /\ UNCHANGED << primary, backup, decision >>

\* Primary makes the decision based on votes.
PrimaryDecide ==
    /\ primary = "alive"
    /\ decision = "none"
    /\ \A p \in Participants : pVote[p] # "none"
    /\ decision' = IF \A p \in Participants : pVote[p] = "yes" THEN "commit" ELSE "abort"
    /\ UNCHANGED << pState, pVote, primary, backup >>

\* Primary crashes after it may or may not have decided.
PrimaryCrash ==
    /\ primary = "alive"
    /\ primary' = "crashed"
    /\ UNCHANGED << pState, pVote, backup, decision >>

\* Backup takes over.  Critically, the backup adopts the existing decision
\* if any has been recorded — never overwriting it (this is what makes
\* recovery safe).
BackupTakeover ==
    /\ primary = "crashed"
    /\ backup  = "standby"
    /\ backup' = "active"
    /\ UNCHANGED << pState, pVote, primary, decision >>

\* If no decision was reached before the crash, the backup decides now.
BackupDecide ==
    /\ backup = "active"
    /\ decision = "none"
    /\ \A p \in Participants : pVote[p] # "none"
    /\ decision' = IF \A p \in Participants : pVote[p] = "yes" THEN "commit" ELSE "abort"
    /\ UNCHANGED << pState, pVote, primary, backup >>

\* Each participant learns the (single) decision.
Learn(p) ==
    /\ decision \in Decisions
    /\ pState[p] = "voted"
    /\ pState' = [pState EXCEPT ![p] = decision]
    /\ UNCHANGED << pVote, primary, backup, decision >>

\* Restart for the next transaction.
Reset ==
    /\ decision \in Decisions
    /\ \A p \in Participants : pState[p] \in Decisions
    /\ pState'   = [p \in Participants |-> "working"]
    /\ pVote'    = [p \in Participants |-> "none"]
    /\ primary'  = "alive"
    /\ backup'   = "standby"
    /\ decision' = "none"

Next == \/ \E p \in Participants : \E v \in {"yes", "no"} : Vote(p, v)
        \/ PrimaryDecide
        \/ PrimaryCrash
        \/ BackupTakeover
        \/ BackupDecide
        \/ \E p \in Participants : Learn(p)
        \/ Reset

Spec == Init /\ [][Next]_vars

TypeOK ==
    /\ pState   \in [Participants -> {"working", "voted", "commit", "abort"}]
    /\ pVote    \in [Participants -> {"none", "yes", "no"}]
    /\ primary  \in {"alive", "crashed"}
    /\ backup   \in {"standby", "active"}
    /\ decision \in {"none", "commit", "abort"}

\* Strong safety: every participant decision matches the recorded global
\* decision, so two participants never see conflicting outcomes.
SafetyInv == \A p \in Participants : pState[p] \in Decisions => pState[p] = decision
====
