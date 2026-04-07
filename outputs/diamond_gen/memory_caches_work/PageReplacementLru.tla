---- MODULE PageReplacementLru ----
(***************************************************************************)
(* LRU page replacement over K frames.                                    *)
(*                                                                         *)
(* The "resident set" is modeled as a sequence ordered from least- to    *)
(* most-recently used.  On a hit the page is moved to the tail (most     *)
(* recently used).  On a miss with a free slot the page is appended; on *)
(* a miss with no free slot the head (LRU) is evicted before appending. *)
(***************************************************************************)
EXTENDS Naturals, Sequences, FiniteSets

CONSTANTS K, Pages

VARIABLE resident   \* sequence of page ids, head = LRU, tail = MRU

vars == << resident >>

PageIds == 1..Pages

\* Removes all occurrences of x from sequence s.
RemoveAll(s, x) ==
    LET F[i \in 0..Len(s)] ==
            IF i = 0 THEN << >>
            ELSE IF s[i] = x THEN F[i-1]
            ELSE Append(F[i-1], s[i])
    IN  F[Len(s)]

Init == resident = << >>

\* Hit: page is currently resident.  Move it to the tail (MRU).
Hit(p) ==
    /\ \E i \in 1..Len(resident) : resident[i] = p
    /\ resident' = Append(RemoveAll(resident, p), p)

\* Miss with free capacity: append at the tail.
MissWithSpace(p) ==
    /\ \A i \in 1..Len(resident) : resident[i] # p
    /\ Len(resident) < K
    /\ resident' = Append(resident, p)

\* Miss when full: evict the head (LRU) and append the new page.
MissEvict(p) ==
    /\ \A i \in 1..Len(resident) : resident[i] # p
    /\ Len(resident) = K
    /\ resident' = Append(Tail(resident), p)

Next == \/ \E p \in PageIds : Hit(p)
        \/ \E p \in PageIds : MissWithSpace(p)
        \/ \E p \in PageIds : MissEvict(p)

Spec == Init /\ [][Next]_vars

\* --- Strong safety properties (folded into TypeOK) ---

\* Bounded capacity: |resident| <= K.
Bounded == Len(resident) \in 0..K

\* No duplicates in the resident set (each page is unique).
NoDuplicates ==
    \A i, j \in 1..Len(resident) : i # j => resident[i] # resident[j]

TypeOK == /\ resident \in Seq(PageIds)
          /\ Bounded
          /\ NoDuplicates
====
