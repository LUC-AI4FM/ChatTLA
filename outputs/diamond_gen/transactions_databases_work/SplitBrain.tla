---- MODULE SplitBrain ----
(***************************************************************************)
(*  Two-replica system with a witness ("tie-breaker").  A replica may   *)
(*  serve writes only when paired with the witness; if the witness is  *)
(*  unreachable, neither replica may serve.  This is the standard      *)
(*  split-brain prevention pattern.                                     *)
(*                                                                         *)
(*  Strong invariant: at most one writer at any time; never two        *)
(*  concurrent writers.                                                 *)
(***************************************************************************)
EXTENDS Naturals, FiniteSets

CONSTANTS Replicas

VARIABLES writer, witnessSeen

vars == << writer, witnessSeen >>

NONE == "none"

Init == /\ writer      = NONE
        /\ witnessSeen = {}

\* A replica acquires the witness link.  Mutual exclusion at the
\* witness: only one replica may be paired with it at a time.
SeeWitness(r) == /\ r \notin witnessSeen
                 /\ witnessSeen = {}
                 /\ witnessSeen' = {r}
                 /\ UNCHANGED writer

\* The witness can be lost only if no replica currently holds the writer
\* role on the witness link.
LoseWitness(r) == /\ r \in witnessSeen
                  /\ writer # r
                  /\ witnessSeen' = witnessSeen \ {r}
                  /\ UNCHANGED writer

\* A replica becomes the (unique) writer iff it has the witness AND no
\* other replica currently does.
BecomeWriter(r) == /\ r \in witnessSeen
                   /\ Cardinality(witnessSeen) = 1
                   /\ writer = NONE
                   /\ writer' = r
                   /\ UNCHANGED witnessSeen

StepDown(r) == /\ writer = r
               /\ writer' = NONE
               /\ UNCHANGED witnessSeen

Next == \/ \E r \in Replicas : SeeWitness(r)
        \/ \E r \in Replicas : LoseWitness(r)
        \/ \E r \in Replicas : BecomeWriter(r)
        \/ \E r \in Replicas : StepDown(r)

Spec == Init /\ [][Next]_vars /\ WF_vars(Next)

\* Strong invariant: at most one writer; the writer (if any) holds the
\* witness; the witness is held by exactly one replica when there is a
\* writer.
TypeOK == /\ writer      \in Replicas \cup {NONE}
          /\ witnessSeen \subseteq Replicas
          /\ (writer # NONE) => (writer \in witnessSeen)
          /\ (writer # NONE) => (Cardinality(witnessSeen) = 1)
====
