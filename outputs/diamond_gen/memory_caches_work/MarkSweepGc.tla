---- MODULE MarkSweepGc ----
(***************************************************************************)
(* Mark-and-sweep GC over a tiny fixed object graph.                      *)
(*                                                                         *)
(* Objects 1..N have a fixed reference structure: each non-leaf points    *)
(* at the next-lower object (1 -> nothing).  Object 1 is the root.       *)
(* Marking starts from the root, follows pointers transitively, and      *)
(* sweep frees everything not marked.                                    *)
(*                                                                         *)
(* The model also lets us "drop" the root edge to a high object so that *)
(* unreachable objects appear and get swept.                             *)
(***************************************************************************)
EXTENDS Naturals, FiniteSets

CONSTANT NumObjects

VARIABLES marked,    \* set of currently-marked objects
          freed,     \* set of swept (freed) objects
          phase,     \* "mark" / "sweep" / "idle"
          rootEdge   \* highest object the root currently points at

vars == << marked, freed, phase, rootEdge >>

Objects == 1..NumObjects

\* Static reference graph: object i's outgoing edge goes to i-1 (chain).
Edges(i) == IF i > 1 THEN {i - 1} ELSE {}

\* Reachable set from current root edge (computed transitively).
Reachable ==
    LET R[k \in 0..NumObjects] ==
            IF k = 0 THEN {rootEdge}
            ELSE R[k-1] \cup
                 UNION { Edges(o) : o \in R[k-1] }
    IN  R[NumObjects]

Init == /\ marked   = {}
        /\ freed    = {}
        /\ phase    = "idle"
        /\ rootEdge = NumObjects

\* Begin marking: snapshot the reachable set, then transition to mark.
StartMark ==
    /\ phase = "idle"
    /\ marked' = Reachable
    /\ phase'  = "sweep"
    /\ UNCHANGED << freed, rootEdge >>

\* Sweep: free every object that is not marked and not already freed.
Sweep ==
    /\ phase = "sweep"
    /\ freed' = freed \cup (Objects \ marked)
    /\ phase' = "idle"
    /\ UNCHANGED << marked, rootEdge >>

\* The mutator may shrink the root edge to a smaller object, making
\* objects above it unreachable.  Models program activity between GCs.
DropRoot ==
    /\ phase = "idle"
    /\ rootEdge > 1
    /\ rootEdge' = rootEdge - 1
    /\ marked'   = {}    \* discard the previous mark set
    /\ UNCHANGED << freed, phase >>

Idle == /\ phase = "idle"
        /\ rootEdge = 1
        /\ UNCHANGED vars

Next == StartMark \/ Sweep \/ DropRoot \/ Idle

Spec == Init /\ [][Next]_vars

\* --- Strong safety properties (folded into TypeOK) ---

\* Mark-sweep contract:
\* (1) Any reachable object is never freed.
\* (2) After marking, every reachable object is in the mark set.
ReachableSurvives ==
    \A o \in Reachable : o \notin freed

MarkCoversReachableAfterMark ==
    phase = "sweep" => Reachable \subseteq marked

TypeOK == /\ marked   \subseteq Objects
          /\ freed    \subseteq Objects
          /\ phase    \in {"mark", "sweep", "idle"}
          /\ rootEdge \in Objects
          /\ ReachableSurvives
          /\ MarkCoversReachableAfterMark
====
