---- MODULE TlbShootdown ----
(***************************************************************************)
(* TLB shootdown across N cores.                                          *)
(*                                                                         *)
(* The initiator core broadcasts an invalidation request.  Every other    *)
(* core must explicitly acknowledge after invalidating its TLB before    *)
(* the shootdown is considered complete.  Until then the initiator       *)
(* spins.  This models the contract: a completed shootdown implies every *)
(* TLB has invalidated.                                                   *)
(***************************************************************************)
EXTENDS Naturals, FiniteSets

CONSTANT Cores

VARIABLES tlbValid,    \* tlbValid[c] = TRUE iff core c has the entry cached
          requested,   \* TRUE iff a shootdown is in progress
          acked,       \* set of cores that have acked
          completed    \* TRUE iff the shootdown has completed

vars == << tlbValid, requested, acked, completed >>

Init == /\ tlbValid  = [c \in Cores |-> TRUE]
        /\ requested = FALSE
        /\ acked     = {}
        /\ completed = FALSE

\* Initiate a shootdown.  Only when no shootdown in progress and not
\* yet completed.  This abstracts a single shootdown round.
Initiate ==
    /\ ~requested
    /\ ~completed
    /\ requested' = TRUE
    /\ UNCHANGED << tlbValid, acked, completed >>

\* A core invalidates its TLB and acknowledges.  Idempotent: if already
\* acked, no further action.
InvalidateAndAck(c) ==
    /\ requested
    /\ ~completed
    /\ c \notin acked
    /\ tlbValid' = [tlbValid EXCEPT ![c] = FALSE]
    /\ acked'    = acked \cup {c}
    /\ UNCHANGED << requested, completed >>

\* Complete the shootdown only when *every* core has acked.
Complete ==
    /\ requested
    /\ ~completed
    /\ acked = Cores
    /\ completed' = TRUE
    /\ UNCHANGED << tlbValid, requested, acked >>

\* Once completed the system idles.
Idle == /\ completed
        /\ UNCHANGED vars

Next == \/ Initiate
        \/ \E c \in Cores : InvalidateAndAck(c)
        \/ Complete
        \/ Idle

Spec == Init /\ [][Next]_vars

\* --- Strong safety properties (folded into TypeOK) ---

\* The defining property of TLB shootdown:
\* if the shootdown is complete then EVERY TLB is invalidated.
CompletedImpliesAllInvalid ==
    completed => \A c \in Cores : ~tlbValid[c]

\* Acked cores have all invalidated their TLBs.
AckedImpliesInvalid ==
    \A c \in acked : ~tlbValid[c]

TypeOK == /\ tlbValid  \in [Cores -> BOOLEAN]
          /\ requested \in BOOLEAN
          /\ acked     \subseteq Cores
          /\ completed \in BOOLEAN
          /\ CompletedImpliesAllInvalid
          /\ AckedImpliesInvalid
====
