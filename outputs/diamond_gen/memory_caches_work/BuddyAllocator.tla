---- MODULE BuddyAllocator ----
(***************************************************************************)
(* Simplified buddy allocator with two size classes: large (1 block of   *)
(* size 2) and small (2 blocks of size 1).  The arena is a single large *)
(* block that can be split into two small buddies and coalesced back.    *)
(*                                                                         *)
(* Each "block" tracks its state: "free", "used", or "split".  The      *)
(* arena's two halves are addressed as positions 0 and 1.                *)
(***************************************************************************)
EXTENDS Naturals, FiniteSets

VARIABLES large,         \* state of the large block: "free" / "used" / "split"
          small          \* small[i] : state of half i in {"free","used"}
                         \*           (only meaningful when large = "split")

vars == << large, small >>

Halves == 0..1

Init == /\ large = "free"
        /\ small = [i \in Halves |-> "free"]

\* Allocate the entire large block (only when free and not split).
AllocLarge ==
    /\ large = "free"
    /\ large' = "used"
    /\ UNCHANGED small

\* Free the entire large block (returns to free).
FreeLarge ==
    /\ large = "used"
    /\ large' = "free"
    /\ UNCHANGED small

\* Split the free large block into two free small buddies.
Split ==
    /\ large = "free"
    /\ large' = "split"
    /\ small' = [i \in Halves |-> "free"]

\* Coalesce two free small buddies back into one free large block.
Coalesce ==
    /\ large = "split"
    /\ \A i \in Halves : small[i] = "free"
    /\ large' = "free"
    /\ small' = [i \in Halves |-> "free"]

\* Allocate one small half (the large must be in the split state).
AllocSmall(i) ==
    /\ large = "split"
    /\ small[i] = "free"
    /\ small' = [small EXCEPT ![i] = "used"]
    /\ UNCHANGED large

\* Free a previously-allocated small half.
FreeSmall(i) ==
    /\ large = "split"
    /\ small[i] = "used"
    /\ small' = [small EXCEPT ![i] = "free"]
    /\ UNCHANGED large

Next == \/ AllocLarge \/ FreeLarge
        \/ Split \/ Coalesce
        \/ \E i \in Halves : AllocSmall(i)
        \/ \E i \in Halves : FreeSmall(i)

Spec == Init /\ [][Next]_vars

\* --- Strong safety properties (folded into TypeOK) ---

\* Buddy invariant: when the large block is NOT split, the small halves
\* must be in their default "free" state (they don't exist as real
\* allocations, so they are not considered "used").  Equivalently:
\* "used" small buddies only exist while the parent is split.
SmallOnlyWhenSplit ==
    \A i \in Halves :
        small[i] = "used" => large = "split"

\* Cannot have a large allocation while halves exist as separately
\* tracked allocations: if large is "used" the splits are not in use.
NoOverlap ==
    large = "used" => \A i \in Halves : small[i] = "free"

TypeOK == /\ large \in {"free", "used", "split"}
          /\ small \in [Halves -> {"free", "used"}]
          /\ SmallOnlyWhenSplit
          /\ NoOverlap
====
