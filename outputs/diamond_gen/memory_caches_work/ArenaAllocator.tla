---- MODULE ArenaAllocator ----
(***************************************************************************)
(* Bump-pointer arena allocator with reset.                               *)
(*                                                                         *)
(* The arena's bump pointer starts at 0.  An allocation of size s         *)
(* succeeds when ptr + s <= ArenaSize and advances the pointer.  Reset   *)
(* sets the pointer back to 0.                                           *)
(***************************************************************************)
EXTENDS Naturals

CONSTANTS ArenaSize, MaxAllocSize

VARIABLES ptr,         \* current bump pointer
          allocCount   \* number of allocations since last reset

vars == << ptr, allocCount >>

Init == /\ ptr        = 0
        /\ allocCount = 0

\* Allocate a block of size s; succeeds only if it fits.
Alloc(s) ==
    /\ s \in 1..MaxAllocSize
    /\ ptr + s <= ArenaSize
    /\ ptr'        = ptr + s
    /\ allocCount' = allocCount + 1

\* Reset wipes the arena: pointer back to 0, alloc count cleared.
Reset ==
    /\ ptr'        = 0
    /\ allocCount' = 0

Next == \/ \E s \in 1..MaxAllocSize : Alloc(s)
        \/ Reset

Spec == Init /\ [][Next]_vars

\* --- Strong safety properties (folded into TypeOK) ---

\* The fundamental bump-allocator bound: pointer is always within arena.
PtrInRange == ptr \in 0..ArenaSize

\* If anything has been allocated then ptr is positive (non-trivial).
NonZeroIfAlloc == (allocCount > 0) => (ptr > 0)

TypeOK == /\ ptr        \in 0..ArenaSize
          /\ allocCount \in Nat
          /\ PtrInRange
          /\ NonZeroIfAlloc
====
