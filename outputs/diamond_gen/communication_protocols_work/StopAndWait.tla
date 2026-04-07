---- MODULE StopAndWait ----
(***************************************************************************)
(* Stop-and-wait protocol over a possibly-duplicating channel.             *)
(* Sender transmits a single tagged frame and blocks until matching ack.   *)
(* Receiver tags acks with the bit of the last accepted frame; duplicate   *)
(* frames are simply re-acked.                                             *)
(* Strong safety: each value is delivered exactly once and in order.       *)
(***************************************************************************)
EXTENDS Naturals, Sequences

MaxSeq == 3

VARIABLES sBit, sIndex, rBit, msgs, acks, delivered

vars == << sBit, sIndex, rBit, msgs, acks, delivered >>

Init ==
    /\ sBit = 0
    /\ sIndex = 0
    /\ rBit = 0
    /\ msgs = {}
    /\ acks = {}
    /\ delivered = << >>

\* Sender (re)transmits the current frame; channel may duplicate.
SendOrRetx ==
    /\ sIndex < MaxSeq
    /\ msgs' = msgs \cup {<<sBit, sIndex>>}
    /\ UNCHANGED << sBit, sIndex, rBit, acks, delivered >>

\* Channel duplicates an in-flight frame (still a set, but action is real).
DupMsg ==
    /\ \E m \in msgs : msgs' = msgs \cup {m}
    /\ UNCHANGED << sBit, sIndex, rBit, acks, delivered >>

\* Receiver delivers a fresh frame and emits matching ack.
RecvFresh ==
    /\ \E m \in msgs :
         /\ m[1] = rBit
         /\ delivered' = Append(delivered, m[2])
         /\ rBit' = 1 - rBit
         /\ acks' = acks \cup {m[1]}
    /\ UNCHANGED << sBit, sIndex, msgs >>

\* Receiver re-acks a stale duplicate without delivering.
RecvDup ==
    /\ \E m \in msgs :
         /\ m[1] # rBit
         /\ acks' = acks \cup {m[1]}
    /\ UNCHANGED << sBit, sIndex, rBit, msgs, delivered >>

\* Sender consumes matching ack, advances, and clears its old in-flight set.
SenderAdvance ==
    /\ sBit \in acks
    /\ sIndex < MaxSeq
    /\ sBit' = 1 - sBit
    /\ sIndex' = sIndex + 1
    /\ acks' = acks \ {sBit}
    /\ msgs' = { m \in msgs : m[1] # sBit }
    /\ UNCHANGED << rBit, delivered >>

Done ==
    /\ sIndex = MaxSeq
    /\ msgs = {}
    /\ acks = {}
    /\ UNCHANGED vars

Next ==
    \/ SendOrRetx \/ DupMsg \/ RecvFresh \/ RecvDup \/ SenderAdvance \/ Done

Spec == Init /\ [][Next]_vars

\* TypeOK conjoins the strong safety property: delivered is the in-order
\* prefix 0,1,..,Len-1 and never has more entries than have been advanced.
TypeOK ==
    /\ sBit \in {0, 1}
    /\ rBit \in {0, 1}
    /\ sIndex \in 0 .. MaxSeq
    /\ msgs \subseteq ({0, 1} \X (0 .. (MaxSeq - 1)))
    /\ acks \subseteq {0, 1}
    /\ Len(delivered) \in 0 .. MaxSeq
    /\ Len(delivered) <= sIndex + 1
    /\ \A i \in 1 .. Len(delivered) : delivered[i] = i - 1
====
