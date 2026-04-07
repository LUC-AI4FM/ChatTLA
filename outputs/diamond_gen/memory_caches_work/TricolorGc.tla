---- MODULE TricolorGc ----
(***************************************************************************)
(* Tricolor incremental mark-and-sweep over a tiny static object graph.   *)
(*                                                                         *)
(* Each object is colored:                                                *)
(*   white -- candidate for collection                                    *)
(*   gray  -- known reachable, children not yet scanned                   *)
(*   black -- known reachable, children fully scanned                     *)
(*                                                                         *)
(* The graph is a chain: object i points to object i-1 (i > 1).  The     *)
(* root is the highest-numbered object.  The tricolor invariant we      *)
(* maintain: NO black object points at a white object.                   *)
(***************************************************************************)
EXTENDS Naturals, FiniteSets

CONSTANT NumObjects

VARIABLES color   \* color[o] in {"white", "gray", "black"}

vars == << color >>

Objects == 1..NumObjects

\* Static reference graph: i -> i-1 (chain).
Edges(i) == IF i > 1 THEN {i - 1} ELSE {}

\* Root is the top object: it begins gray at the start of marking.
Root == NumObjects

Init == color = [o \in Objects |->
                    IF o = Root THEN "gray" ELSE "white"]

\* Scan a gray object: its children become gray (or stay non-white) and
\* the object itself becomes black.  This is the only way to introduce
\* black-ness, and it always pre-grays children -> tricolor preserved.
Scan(o) ==
    /\ color[o] = "gray"
    /\ color' = [c \in Objects |->
                    IF c = o THEN "black"
                    ELSE IF c \in Edges(o) /\ color[c] = "white" THEN "gray"
                    ELSE color[c]]

\* Idle once nothing is gray (mark phase complete).
Idle == /\ \A o \in Objects : color[o] # "gray"
        /\ UNCHANGED vars

Next == \/ \E o \in Objects : Scan(o)
        \/ Idle

Spec == Init /\ [][Next]_vars

\* --- Strong safety properties (folded into TypeOK) ---

\* The tricolor invariant: no black object points at a white object.
TricolorInvariant ==
    \A b \in Objects :
        color[b] = "black" =>
            \A w \in Edges(b) : color[w] # "white"

\* When idle, every object that the root could reach is non-white
\* (i.e. survived the marking).  Since this is a chain from Root,
\* the entire chain must be non-white once marking has completed.
ReachableMarkedAtFixpoint ==
    (\A o \in Objects : color[o] # "gray") =>
        (\A o \in Objects : color[o] # "white")

TypeOK == /\ color \in [Objects -> {"white", "gray", "black"}]
          /\ TricolorInvariant
          /\ ReachableMarkedAtFixpoint
====
