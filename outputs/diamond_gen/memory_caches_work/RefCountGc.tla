---- MODULE RefCountGc ----
(***************************************************************************)
(* Reference-counted GC over a small object graph.                        *)
(*                                                                         *)
(* A small fixed set of objects has dynamically-managed reference counts.*)
(* A "root" set always points at object 1.  inc/dec maintain the count;  *)
(* an object is freed when its count reaches zero.                       *)
(* Safety: no live (root-reachable) object is ever freed.                *)
(***************************************************************************)
EXTENDS Naturals, FiniteSets

CONSTANTS NumObjects, MaxRefs

VARIABLES rc,        \* rc[o] : reference count of object o (0..MaxRefs)
          freed      \* freed[o] : TRUE iff object o has been freed

vars == << rc, freed >>

Objects == 1..NumObjects

\* Object 1 is the "root" -- always live in this model.
Root == 1

Init == /\ rc    = [o \in Objects |-> IF o = Root THEN 1 ELSE 0]
        /\ freed = [o \in Objects |-> FALSE]

\* Increment a non-freed object's count (cap MaxRefs to keep finite).
Inc(o) == /\ ~freed[o]
          /\ rc[o] < MaxRefs
          /\ rc'    = [rc EXCEPT ![o] = @ + 1]
          /\ UNCHANGED freed

\* Decrement a positive count.  Root is never decremented below 1.
Dec(o) == /\ ~freed[o]
          /\ rc[o] > (IF o = Root THEN 1 ELSE 0)
          /\ rc'    = [rc EXCEPT ![o] = @ - 1]
          /\ UNCHANGED freed

\* Free an object whose count has dropped to zero.
Free(o) == /\ ~freed[o]
           /\ rc[o] = 0
           /\ freed' = [freed EXCEPT ![o] = TRUE]
           /\ UNCHANGED rc

Idle == /\ \A o \in Objects : freed[o] \/ o = Root
        /\ UNCHANGED vars

Next == \/ \E o \in Objects : Inc(o)
        \/ \E o \in Objects : Dec(o)
        \/ \E o \in Objects : Free(o)
        \/ Idle

Spec == Init /\ [][Next]_vars

\* --- Strong safety properties (folded into TypeOK) ---

\* Refcount-GC contract:
\* (1) The root object is never freed.
\* (2) A freed object has refcount zero.
RootNeverFreed == ~freed[Root]
FreedHasZeroRc == \A o \in Objects : freed[o] => rc[o] = 0

TypeOK == /\ rc    \in [Objects -> 0..MaxRefs]
          /\ freed \in [Objects -> BOOLEAN]
          /\ RootNeverFreed
          /\ FreedHasZeroRc
====
