---- MODULE DirectoryCoherence ----
(***************************************************************************)
(* Directory-based cache coherence with one directory and N caches.      *)
(*                                                                         *)
(* For a single cache line, the directory tracks:                        *)
(*   sharers : set of caches currently holding a clean (Shared) copy     *)
(*   owner   : the unique writer (or NONE)                               *)
(* When the line is owned, sharers is empty.  Each cache also has its    *)
(* own state: I, S, or M.                                                *)
(***************************************************************************)
EXTENDS Naturals, FiniteSets

CONSTANT Caches

NONE == "none"

VARIABLES sharers,    \* set of cores holding S
          owner,      \* core holding M, or NONE
          state       \* state[c] : local state of cache c

vars == << sharers, owner, state >>

States == {"I", "S", "M"}

Init == /\ sharers = {}
        /\ owner   = NONE
        /\ state   = [c \in Caches |-> "I"]

\* Cache c reads: must be I, no current writer.
GetShared(c) ==
    /\ state[c] = "I"
    /\ owner = NONE
    /\ sharers' = sharers \cup {c}
    /\ state'   = [state EXCEPT ![c] = "S"]
    /\ UNCHANGED owner

\* Cache c reads while another cache owns: directory invalidates the
\* owner; ownership is downgraded to shared.
GetSharedFromOwner(c) ==
    /\ state[c] = "I"
    /\ owner # NONE
    /\ owner # c
    /\ sharers' = {c, owner}
    /\ state'   = [state EXCEPT ![c] = "S", ![owner] = "S"]
    /\ owner'   = NONE

\* Cache c writes: invalidate every other cache; c becomes the owner.
GetModified(c) ==
    /\ state[c] # "M"
    /\ owner'   = c
    /\ sharers' = {}
    /\ state'   = [d \in Caches |->
                       IF d = c THEN "M" ELSE "I"]

\* Eviction: a Shared cache silently drops the line.
EvictShared(c) ==
    /\ state[c] = "S"
    /\ sharers' = sharers \ {c}
    /\ state'   = [state EXCEPT ![c] = "I"]
    /\ UNCHANGED owner

\* Eviction with writeback: the owner returns the line.
EvictModified(c) ==
    /\ state[c] = "M"
    /\ owner = c
    /\ owner'   = NONE
    /\ state'   = [state EXCEPT ![c] = "I"]
    /\ UNCHANGED sharers

Next == \/ \E c \in Caches : GetShared(c)
        \/ \E c \in Caches : GetSharedFromOwner(c)
        \/ \E c \in Caches : GetModified(c)
        \/ \E c \in Caches : EvictShared(c)
        \/ \E c \in Caches : EvictModified(c)

Spec == Init /\ [][Next]_vars

\* --- Strong safety properties (folded into TypeOK) ---

\* At most one writer (owner is unique).
SingleOwner ==
    Cardinality({c \in Caches : state[c] = "M"}) <= 1

\* The owner field agrees with cache states.
OwnerAgreement ==
    /\ (owner # NONE) => (state[owner] = "M")
    /\ (owner = NONE) => (\A c \in Caches : state[c] # "M")

\* When owned, no cache is in Shared state and the directory's sharers
\* set is empty.
OwnerExcludesShared ==
    (owner # NONE) => (sharers = {} /\ \A c \in Caches : state[c] # "S")

\* Sharers in the directory are exactly the caches in S state.
SharersAgreement ==
    sharers = {c \in Caches : state[c] = "S"}

TypeOK == /\ sharers \subseteq Caches
          /\ owner   \in Caches \cup {NONE}
          /\ state   \in [Caches -> States]
          /\ SingleOwner
          /\ OwnerAgreement
          /\ OwnerExcludesShared
          /\ SharersAgreement
====
