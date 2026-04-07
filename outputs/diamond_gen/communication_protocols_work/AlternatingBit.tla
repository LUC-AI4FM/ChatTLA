---- MODULE AlternatingBit ----
(***************************************************************************)
(* Alternating-bit protocol over a lossy (no-duplicating, no-reordering)   *)
(* channel.  Sender alternates a tag bit; receiver acks the bit it last    *)
(* accepted.  Strong safety: the sequence the receiver has delivered is    *)
(* always a prefix of the sequence the sender has produced (in-order, no   *)
(* duplication, no fabrication).                                           *)
(***************************************************************************)
EXTENDS Naturals, Sequences, FiniteSets

MaxSeq == 3
Vals == 0 .. (MaxSeq - 1)

VARIABLES
    sBit,        \* sender's current alternating bit
    sIndex,      \* index of next value the sender will produce (0..MaxSeq)
    rBit,        \* bit the receiver expects next
    delivered,   \* sequence of values delivered by the receiver
    msgChan,     \* set of in-flight (bit, value) data messages
    ackChan      \* set of in-flight ack bits

vars == << sBit, sIndex, rBit, delivered, msgChan, ackChan >>

Init ==
    /\ sBit = 0
    /\ sIndex = 0
    /\ rBit = 0
    /\ delivered = << >>
    /\ msgChan = {}
    /\ ackChan = {}

\* Sender (re)transmits the current message tagged with sBit.
Send ==
    /\ sIndex < MaxSeq
    /\ msgChan' = msgChan \cup {<<sBit, sIndex>>}
    /\ Cardinality(msgChan) < 3
    /\ UNCHANGED << sBit, sIndex, rBit, delivered, ackChan >>

\* Channel may drop a data message.
LoseMsg ==
    /\ \E m \in msgChan :
         msgChan' = msgChan \ {m}
    /\ UNCHANGED << sBit, sIndex, rBit, delivered, ackChan >>

\* Channel may drop an ack.
LoseAck ==
    /\ \E a \in ackChan :
         ackChan' = ackChan \ {a}
    /\ UNCHANGED << sBit, sIndex, rBit, delivered, msgChan >>

\* Receiver accepts a message whose bit matches what it expects.
RecvGood ==
    /\ \E m \in msgChan :
         /\ m[1] = rBit
         /\ delivered' = Append(delivered, m[2])
         /\ rBit' = 1 - rBit
         /\ ackChan' = ackChan \cup {m[1]}
         /\ UNCHANGED << sBit, sIndex, msgChan >>

\* Receiver re-acks a duplicate (wrong bit) without delivering.
RecvDup ==
    /\ \E m \in msgChan :
         /\ m[1] # rBit
         /\ ackChan' = ackChan \cup {1 - rBit}
         /\ UNCHANGED << sBit, sIndex, rBit, delivered, msgChan >>

\* Sender receives a matching ack and advances.
RecvAck ==
    /\ sBit \in ackChan
    /\ sIndex < MaxSeq
    /\ sBit' = 1 - sBit
    /\ sIndex' = sIndex + 1
    /\ ackChan' = ackChan \ {sBit}
    \* Sender purges its old in-flight messages once acked.
    /\ msgChan' = { m \in msgChan : m[1] # sBit }
    /\ UNCHANGED << rBit, delivered >>

\* Idle stutter once everything has been delivered, to avoid TLC deadlock.
Done ==
    /\ sIndex = MaxSeq
    /\ msgChan = {}
    /\ ackChan = {}
    /\ UNCHANGED vars

Next == \/ Send \/ LoseMsg \/ LoseAck \/ RecvGood \/ RecvDup \/ RecvAck \/ Done

Spec == Init /\ [][Next]_vars

\* Strong safety conjoined into TypeOK so the mutation test sees a real
\* invariant: delivered is exactly the in-order prefix 0,1,..,Len-1, and
\* its length never exceeds the number of values the sender has committed.
InOrderPrefix ==
    /\ Len(delivered) \in 0 .. MaxSeq
    /\ \A i \in 1 .. Len(delivered) : delivered[i] = i - 1
    /\ Len(delivered) <= sIndex + 1

TypeOK ==
    /\ sBit \in {0, 1}
    /\ rBit \in {0, 1}
    /\ sIndex \in 0 .. MaxSeq
    /\ msgChan \subseteq ({0,1} \X Vals)
    /\ ackChan \subseteq {0, 1}
    /\ InOrderPrefix
====
