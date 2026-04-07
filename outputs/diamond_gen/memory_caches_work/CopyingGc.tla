---- MODULE CopyingGc ----
(***************************************************************************)
(* Stop-the-world copying GC with from-space and to-space.                *)
(*                                                                         *)
(* A small pool of objects is initially placed in from-space and may be  *)
(* allocated.  When GC runs it copies every reachable object to          *)
(* to-space and clears from-space.                                       *)
(* Reachability here is modeled by an explicit "rooted" set the mutator  *)
(* maintains; unrooted objects are garbage.                              *)
(***************************************************************************)
EXTENDS Naturals, FiniteSets

CONSTANT NumObjects

VARIABLES fromSpace,   \* objects currently in from-space
          toSpace,     \* objects currently in to-space
          rooted,      \* objects the mutator considers reachable
          phase        \* "mutate" / "copy" / "done"

vars == << fromSpace, toSpace, rooted, phase >>

Objects == 1..NumObjects

Init == /\ fromSpace = Objects
        /\ toSpace   = {}
        /\ rooted    = Objects
        /\ phase     = "mutate"

\* Mutator drops the root on some object, making it garbage.
DropRoot(o) ==
    /\ phase = "mutate"
    /\ o \in rooted
    /\ rooted' = rooted \ {o}
    /\ UNCHANGED << fromSpace, toSpace, phase >>

\* GC starts -- snapshot taken at this point.
StartGc ==
    /\ phase = "mutate"
    /\ phase' = "copy"
    /\ UNCHANGED << fromSpace, toSpace, rooted >>

\* Copy ALL reachable objects from from-space to to-space, then empty
\* from-space.  Stop-the-world: a single atomic step.
Copy ==
    /\ phase = "copy"
    /\ toSpace'   = rooted \cap fromSpace
    /\ fromSpace' = {}
    /\ phase'     = "done"
    /\ UNCHANGED rooted

Idle == /\ phase = "done"
        /\ UNCHANGED vars

Next == \/ \E o \in Objects : DropRoot(o)
        \/ StartGc
        \/ Copy
        \/ Idle

Spec == Init /\ [][Next]_vars

\* --- Strong safety properties (folded into TypeOK) ---

\* Copying-GC contract:
\* (1) After copying, every rooted object lives in to-space.
\* (2) After copying, from-space is empty.
DoneImpliesCopied ==
    phase = "done" => rooted \subseteq toSpace

DoneImpliesFromEmpty ==
    phase = "done" => fromSpace = {}

\* From-space and to-space are always disjoint -- an object cannot be
\* in both spaces at once.
SpacesDisjoint == fromSpace \cap toSpace = {}

TypeOK == /\ fromSpace \subseteq Objects
          /\ toSpace   \subseteq Objects
          /\ rooted    \subseteq Objects
          /\ phase     \in {"mutate", "copy", "done"}
          /\ DoneImpliesCopied
          /\ DoneImpliesFromEmpty
          /\ SpacesDisjoint
====
