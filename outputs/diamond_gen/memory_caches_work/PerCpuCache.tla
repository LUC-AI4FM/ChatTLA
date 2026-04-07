---- MODULE PerCpuCache ----
(***************************************************************************)
(* Slab-style per-CPU object cache backed by a global pool.               *)
(*                                                                         *)
(* Each CPU has a private cache of capacity <= CacheCap.  When the local *)
(* cache is empty a "refill" pulls Batch objects from the global pool;   *)
(* when it is full a "drain" pushes Batch objects back.  Allocation     *)
(* and free operate purely on the local cache.                          *)
(*                                                                         *)
(* Conservation: total objects across all caches plus the pool plus     *)
(* the in-use count is constant.                                         *)
(***************************************************************************)
EXTENDS Naturals, FiniteSets

CONSTANTS Cpus, TotalObjects, CacheCap, Batch

VARIABLES cache,    \* cache[c] : count of objects in CPU c's cache
          pool,     \* count of objects in the global pool
          inuse     \* count of objects currently allocated to clients

vars == << cache, pool, inuse >>

Init == /\ cache = [c \in Cpus |-> 0]
        /\ pool  = TotalObjects
        /\ inuse = 0

\* Refill: empty cache pulls Batch from pool (if pool has enough).
Refill(c) ==
    /\ cache[c] = 0
    /\ pool >= Batch
    /\ cache' = [cache EXCEPT ![c] = Batch]
    /\ pool'  = pool - Batch
    /\ UNCHANGED inuse

\* Drain: full cache pushes Batch back to pool.
Drain(c) ==
    /\ cache[c] = CacheCap
    /\ cache' = [cache EXCEPT ![c] = CacheCap - Batch]
    /\ pool'  = pool + Batch
    /\ UNCHANGED inuse

\* Allocate one object out of CPU c's local cache.
Alloc(c) ==
    /\ cache[c] > 0
    /\ cache' = [cache EXCEPT ![c] = @ - 1]
    /\ inuse' = inuse + 1
    /\ UNCHANGED pool

\* Free one object back to CPU c's local cache.
Free(c) ==
    /\ inuse > 0
    /\ cache[c] < CacheCap
    /\ cache' = [cache EXCEPT ![c] = @ + 1]
    /\ inuse' = inuse - 1
    /\ UNCHANGED pool

Next == \/ \E c \in Cpus : Refill(c)
        \/ \E c \in Cpus : Drain(c)
        \/ \E c \in Cpus : Alloc(c)
        \/ \E c \in Cpus : Free(c)

Spec == Init /\ [][Next]_vars

\* Sum of all per-CPU cache occupancies.
TotalCached == LET S[X \in SUBSET Cpus] ==
                    IF X = {} THEN 0
                    ELSE LET c == CHOOSE x \in X : TRUE
                         IN  cache[c] + S[X \ {c}]
               IN  S[Cpus]

\* --- Strong safety properties (folded into TypeOK) ---

\* Conservation: nothing is ever created or destroyed.
Conservation == TotalCached + pool + inuse = TotalObjects

\* Each cache occupancy stays within its capacity.
CacheBounded == \A c \in Cpus : cache[c] \in 0..CacheCap

TypeOK == /\ cache \in [Cpus -> 0..CacheCap]
          /\ pool  \in 0..TotalObjects
          /\ inuse \in 0..TotalObjects
          /\ Conservation
          /\ CacheBounded
====
