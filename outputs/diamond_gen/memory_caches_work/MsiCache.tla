---- MODULE MsiCache ----
(***************************************************************************)
(* MSI cache coherence protocol with two cores and a single cache line.    *)
(* Each core's local copy of the line is in one of three states:           *)
(*   M (Modified) - core owns the line, may write, others Invalid          *)
(*   S (Shared)   - read-only copy; other cores may also be Shared         *)
(*   I (Invalid)  - no valid copy                                          *)
(***************************************************************************)
EXTENDS Naturals, FiniteSets

CONSTANT Cores

VARIABLE state    \* state[c] in {"M","S","I"}

vars == << state >>

States == {"M", "S", "I"}

Init == state = [c \in Cores |-> "I"]

\* Bus read: a core in I asks for a copy.
\* If any other core is in M, that core downgrades to S (writeback).
\* All other valid copies remain S; requester becomes S.
BusRead(c) ==
    /\ state[c] = "I"
    /\ state' = [d \in Cores |->
                    IF d = c THEN "S"
                    ELSE IF state[d] = "M" THEN "S"
                    ELSE state[d]]

\* Bus read-for-ownership: a core wants to write.
\* All other cores' copies are invalidated; requester becomes M.
BusReadX(c) ==
    /\ state[c] # "M"
    /\ state' = [d \in Cores |->
                    IF d = c THEN "M" ELSE "I"]

\* Local write hit on a Shared line upgrades to M, invalidating others.
WriteHit(c) ==
    /\ state[c] = "S"
    /\ state' = [d \in Cores |->
                    IF d = c THEN "M" ELSE "I"]

\* A Modified line may be silently evicted (writeback) back to Invalid.
Evict(c) ==
    /\ state[c] # "I"
    /\ state' = [state EXCEPT ![c] = "I"]

Next == \/ \E c \in Cores : BusRead(c)
        \/ \E c \in Cores : BusReadX(c)
        \/ \E c \in Cores : WriteHit(c)
        \/ \E c \in Cores : Evict(c)

Spec == Init /\ [][Next]_vars

\* --- Strong safety properties (folded into TypeOK) ---

\* At most one core may hold the line in Modified state.
SingleWriter == Cardinality({c \in Cores : state[c] = "M"}) <= 1

\* If any core is Modified, no other core may be Shared.
ModifiedExcludesShared ==
    \A c \in Cores : state[c] = "M" =>
        \A d \in Cores \ {c} : state[d] = "I"

TypeOK == /\ state \in [Cores -> States]
          /\ SingleWriter
          /\ ModifiedExcludesShared
====
