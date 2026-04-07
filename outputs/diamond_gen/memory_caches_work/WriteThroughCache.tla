---- MODULE WriteThroughCache ----
(***************************************************************************)
(* Write-through cache: every write hits both the cache and main memory  *)
(* atomically.  A miss simply loads memory into the cache; an eviction  *)
(* drops the cached value.                                              *)
(*                                                                         *)
(* Safety: whenever the cache holds a value, it equals memory's value.  *)
(***************************************************************************)
EXTENDS Naturals

CONSTANT MaxVal

VARIABLES mem,         \* main memory value
          cached,      \* cache value (only meaningful when valid)
          valid        \* TRUE iff the cache currently holds a value

vars == << mem, cached, valid >>

Vals == 0..MaxVal

Init == /\ mem    = 0
        /\ cached = 0
        /\ valid  = FALSE

\* Read-miss: load memory into the cache.
LoadFromMemory ==
    /\ ~valid
    /\ cached' = mem
    /\ valid'  = TRUE
    /\ UNCHANGED mem

\* Read-hit: trivial; no state change.
ReadHit ==
    /\ valid
    /\ UNCHANGED vars

\* Write-through: write goes to BOTH cache and memory atomically.
\* If the cache wasn't holding the line we install it.
Write(v) ==
    /\ v \in Vals
    /\ mem'    = v
    /\ cached' = v
    /\ valid'  = TRUE

\* Eviction: drop the cached copy (clean -- nothing to write back).
Evict ==
    /\ valid
    /\ valid'  = FALSE
    /\ UNCHANGED << mem, cached >>

Next == \/ LoadFromMemory
        \/ ReadHit
        \/ \E v \in Vals : Write(v)
        \/ Evict

Spec == Init /\ [][Next]_vars

\* --- Strong safety properties (folded into TypeOK) ---

\* The defining property of write-through:
\* whenever the cache is valid, the cached value equals memory.
CacheCoherent == valid => cached = mem

TypeOK == /\ mem    \in Vals
          /\ cached \in Vals
          /\ valid  \in BOOLEAN
          /\ CacheCoherent
====
