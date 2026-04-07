---- MODULE MesiCache ----
(***************************************************************************)
(* MESI cache coherence protocol with two cores and a single line.         *)
(* States:                                                                 *)
(*   M (Modified)  - dirty, sole owner, may write silently                 *)
(*   E (Exclusive) - clean, sole owner, may upgrade to M silently          *)
(*   S (Shared)    - clean, possibly cached by others                      *)
(*   I (Invalid)   - no copy                                               *)
(* The protocol distinguishes E from M so a write to E does NOT need a    *)
(* bus transaction.                                                        *)
(***************************************************************************)
EXTENDS Naturals, FiniteSets

CONSTANT Cores

VARIABLE state    \* state[c] in {"M","E","S","I"}

vars == << state >>

States == {"M", "E", "S", "I"}

Init == state = [c \in Cores |-> "I"]

OtherValid(c) == \E d \in Cores \ {c} : state[d] # "I"

\* Bus read on a miss.
\* If no other core has the line, requester becomes Exclusive.
\* Otherwise the requester is Shared, any Modified owner writebacks to S,
\* and any Exclusive owner downgrades to S.
BusRead(c) ==
    /\ state[c] = "I"
    /\ state' = [d \in Cores |->
                    IF d = c
                       THEN IF OtherValid(c) THEN "S" ELSE "E"
                    ELSE IF state[d] \in {"M","E"} THEN "S"
                    ELSE state[d]]

\* Bus read-for-ownership: write miss / write to Shared. Invalidates others.
BusReadX(c) ==
    /\ state[c] \in {"I", "S"}
    /\ state' = [d \in Cores |->
                    IF d = c THEN "M" ELSE "I"]

\* Silent upgrade: Exclusive -> Modified on write hit (no bus traffic).
SilentUpgrade(c) ==
    /\ state[c] = "E"
    /\ state' = [state EXCEPT ![c] = "M"]

\* Silent eviction (writeback if M).
Evict(c) ==
    /\ state[c] # "I"
    /\ state' = [state EXCEPT ![c] = "I"]

Next == \/ \E c \in Cores : BusRead(c)
        \/ \E c \in Cores : BusReadX(c)
        \/ \E c \in Cores : SilentUpgrade(c)
        \/ \E c \in Cores : Evict(c)

Spec == Init /\ [][Next]_vars

\* --- Strong safety properties (folded into TypeOK) ---

\* At most one Modified copy across all cores.
SingleModified == Cardinality({c \in Cores : state[c] = "M"}) <= 1

\* At most one Exclusive copy across all cores.
SingleExclusive == Cardinality({c \in Cores : state[c] = "E"}) <= 1

\* M and E are mutually exclusive across cores: an owned line is owned
\* by exactly one core, in exactly one of {M, E}.
OwnerExclusive ==
    \A c \in Cores : state[c] \in {"M","E"} =>
        \A d \in Cores \ {c} : state[d] = "I"

TypeOK == /\ state \in [Cores -> States]
          /\ SingleModified
          /\ SingleExclusive
          /\ OwnerExclusive
====
