---- MODULE PageReplacementClock ----
(***************************************************************************)
(* Clock (second-chance) page replacement over K frames.                  *)
(*                                                                         *)
(* Each frame holds either NONE or a page id, plus a reference bit.       *)
(* On a miss with no free frame the clock hand advances; if the frame    *)
(* under the hand has ref=0 it is evicted, otherwise ref is cleared and  *)
(* the hand moves on.  This faithfully models the classic algorithm.     *)
(***************************************************************************)
EXTENDS Naturals, FiniteSets

CONSTANTS K, Pages

NONE == 0   \* sentinel page-id meaning "frame is empty"

VARIABLES frame,    \* frame[i] : page id in slot i, or NONE
          ref,      \* ref[i]   : reference bit (0 or 1)
          hand      \* clock hand : 0..K-1

vars == << frame, ref, hand >>

Frames == 0..(K - 1)
PageIds == 1..Pages

Init == /\ frame = [i \in Frames |-> NONE]
        /\ ref   = [i \in Frames |-> 0]
        /\ hand  = 0

\* Already-resident hit: refresh the reference bit.
Hit(p) ==
    /\ \E i \in Frames : frame[i] = p
    /\ ref' = [i \in Frames |-> IF frame[i] = p THEN 1 ELSE ref[i]]
    /\ UNCHANGED << frame, hand >>

\* Miss with a free frame available: place the page, set ref=1.
MissFreeSlot(p) ==
    /\ \A i \in Frames : frame[i] # p
    /\ \E i \in Frames : frame[i] = NONE
    /\ LET f == CHOOSE i \in Frames : frame[i] = NONE
       IN  /\ frame' = [frame EXCEPT ![f] = p]
           /\ ref'   = [ref   EXCEPT ![f] = 1]
           /\ UNCHANGED hand

\* Miss with no free slot: clock hand advances.  If frame under hand has
\* ref=0 it is evicted and replaced.  Otherwise ref is cleared and hand
\* moves on (one step).
MissEvict(p) ==
    /\ \A i \in Frames : frame[i] # p
    /\ \A i \in Frames : frame[i] # NONE
    /\ IF ref[hand] = 0
          THEN /\ frame' = [frame EXCEPT ![hand] = p]
               /\ ref'   = [ref   EXCEPT ![hand] = 1]
               /\ hand'  = (hand + 1) % K
          ELSE /\ ref'  = [ref EXCEPT ![hand] = 0]
               /\ hand' = (hand + 1) % K
               /\ UNCHANGED frame

Next == \/ \E p \in PageIds : Hit(p)
        \/ \E p \in PageIds : MissFreeSlot(p)
        \/ \E p \in PageIds : MissEvict(p)

Spec == Init /\ [][Next]_vars

\* --- Strong safety properties (folded into TypeOK) ---

\* The set of resident pages.
Resident == {frame[i] : i \in Frames} \ {NONE}

\* At most K resident pages -- the canonical page-replacement bound.
ResidentBounded == Cardinality(Resident) <= K

\* No frame holds the same page twice.
NoDuplicates ==
    \A i, j \in Frames :
        (i # j /\ frame[i] # NONE) => frame[i] # frame[j]

TypeOK == /\ frame \in [Frames -> (PageIds \cup {NONE})]
          /\ ref   \in [Frames -> 0..1]
          /\ hand  \in Frames
          /\ ResidentBounded
          /\ NoDuplicates
====
