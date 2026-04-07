---- MODULE FreeList ----
(***************************************************************************)
(* Free-list allocator over K fixed-size blocks.                          *)
(*                                                                         *)
(* Each block is in exactly one of two sets at any moment: "free" or     *)
(* "used".  malloc removes a block from free and adds it to used; free  *)
(* does the reverse.  The fundamental invariant is the partition.       *)
(***************************************************************************)
EXTENDS Naturals, FiniteSets

CONSTANT NumBlocks

VARIABLES freeBlocks, usedBlocks

vars == << freeBlocks, usedBlocks >>

Blocks == 1..NumBlocks

Init == /\ freeBlocks = Blocks
        /\ usedBlocks = {}

\* malloc: pick any free block and mark it used.
Malloc ==
    /\ freeBlocks # {}
    /\ \E b \in freeBlocks :
           /\ freeBlocks' = freeBlocks \ {b}
           /\ usedBlocks' = usedBlocks \cup {b}

\* free: pick any used block and return it to the free list.
FreeBlock ==
    /\ usedBlocks # {}
    /\ \E b \in usedBlocks :
           /\ usedBlocks' = usedBlocks \ {b}
           /\ freeBlocks' = freeBlocks \cup {b}

Next == Malloc \/ FreeBlock

Spec == Init /\ [][Next]_vars

\* --- Strong safety properties (folded into TypeOK) ---

\* Partition invariant: every block is in exactly one set.
Partition ==
    /\ freeBlocks \cup usedBlocks = Blocks
    /\ freeBlocks \cap usedBlocks = {}

\* The two sets together always cover the full block range.
SizeBound == Cardinality(freeBlocks) + Cardinality(usedBlocks) = NumBlocks

TypeOK == /\ freeBlocks \subseteq Blocks
          /\ usedBlocks \subseteq Blocks
          /\ Partition
          /\ SizeBound
====
