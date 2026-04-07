---- MODULE WitnessReplication ----
(***************************************************************************)
(* Primary-backup replication with a witness replica.  The witness holds  *)
(* no state but votes to break ties between primary and backup.  A log    *)
(* entry is committed once a majority (primary + backup, or primary +     *)
(* witness) acknowledges it.                                              *)
(* Safety: the committed prefix is identical on every replica in some    *)
(* majority quorum.                                                      *)
(***************************************************************************)
EXTENDS Naturals, FiniteSets

Replicas  == {"primary", "backup", "witness"}
Roles     == {"primary", "backup", "witness"}
MaxLog    == 2

VARIABLES log, committed

vars == << log, committed >>

\* Each replica's log length (the witness only ever stores 0 or has a vote
\* counter, modeled as a length).
Init == /\ log       = [r \in Replicas |-> 0]
        /\ committed = 0

\* The primary appends an entry locally.
Append ==
    /\ log["primary"] < MaxLog
    /\ log' = [log EXCEPT !["primary"] = @ + 1]
    /\ UNCHANGED committed

\* The backup catches up to the primary (replicating the new entry).
Replicate ==
    /\ log["backup"] < log["primary"]
    /\ log' = [log EXCEPT !["backup"] = @ + 1]
    /\ UNCHANGED committed

\* The witness votes for the latest primary log length, breaking ties.
WitnessVote ==
    /\ log["witness"] < log["primary"]
    /\ log' = [log EXCEPT !["witness"] = @ + 1]
    /\ UNCHANGED committed

\* Commit a length once at least two replicas (a majority) hold it.
Commit ==
    /\ committed < log["primary"]
    /\ Cardinality({r \in Replicas : log[r] >= committed + 1}) >= 2
    /\ committed' = committed + 1
    /\ UNCHANGED log

\* Restart for the next batch.
Reset ==
    /\ committed = MaxLog
    /\ log'       = [r \in Replicas |-> 0]
    /\ committed' = 0

Next == \/ Append
        \/ Replicate
        \/ WitnessVote
        \/ Commit
        \/ Reset

Spec == Init /\ [][Next]_vars

TypeOK ==
    /\ log       \in [Replicas -> 0..MaxLog]
    /\ committed \in 0..MaxLog

\* Strong safety: the committed prefix is held by a majority of replicas.
SafetyInv == (committed > 0) => Cardinality({r \in Replicas : log[r] >= committed}) >= 2
====
