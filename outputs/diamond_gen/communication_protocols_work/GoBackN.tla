---- MODULE GoBackN ----
(***************************************************************************)
(* Go-back-N retransmission with window size W.  On a timeout the sender   *)
(* re-sends every unacked frame in [base, next).  The receiver only        *)
(* accepts the frame whose seq number equals "expected"; out-of-order      *)
(* frames are silently dropped.                                            *)
(* Strong safety: the receiver's delivered sequence is a prefix of the     *)
(* sender's input 0,1,...,MaxSeq-1, and the in-flight window stays bounded.*)
(***************************************************************************)
EXTENDS Naturals, Sequences, FiniteSets

W == 2
MaxSeq == 4

VARIABLES base, next, expected, channel, delivered

vars == << base, next, expected, channel, delivered >>

Init ==
    /\ base = 0
    /\ next = 0
    /\ expected = 0
    /\ channel = {}
    /\ delivered = << >>

\* Sender enqueues a fresh frame if window has room.
SendNew ==
    /\ next < MaxSeq
    /\ next - base < W
    /\ channel' = channel \cup {next}
    /\ next' = next + 1
    /\ UNCHANGED << base, expected, delivered >>

\* Timeout: re-send the entire current window.
RetransmitWindow ==
    /\ base < next
    /\ channel' = channel \cup { s \in 0 .. MaxSeq : base <= s /\ s < next }
    /\ UNCHANGED << base, next, expected, delivered >>

\* Channel may drop any frame.
DropFrame ==
    /\ \E s \in channel : channel' = channel \ {s}
    /\ UNCHANGED << base, next, expected, delivered >>

\* Receiver delivers a frame iff it equals expected.
DeliverInOrder ==
    /\ expected \in channel
    /\ delivered' = Append(delivered, expected)
    /\ expected' = expected + 1
    /\ channel' = channel \ {expected}
    /\ UNCHANGED << base, next >>

\* Cumulative ack: sender slides base.
SlideBase ==
    /\ base < expected
    /\ base' = base + 1
    /\ UNCHANGED << next, expected, channel, delivered >>

Done ==
    /\ next = MaxSeq
    /\ expected = MaxSeq
    /\ base = MaxSeq
    /\ UNCHANGED vars

Next ==
    \/ SendNew \/ RetransmitWindow \/ DropFrame
    \/ DeliverInOrder \/ SlideBase \/ Done

Spec == Init /\ [][Next]_vars

\* Strong safety conjoined into TypeOK: prefix delivery, bounded window.
TypeOK ==
    /\ base \in 0 .. MaxSeq
    /\ next \in 0 .. MaxSeq
    /\ expected \in 0 .. MaxSeq
    /\ channel \subseteq 0 .. (MaxSeq - 1)
    /\ Len(delivered) \in 0 .. MaxSeq
    /\ base <= expected
    /\ expected <= next
    /\ next - base <= W
    /\ Len(delivered) = expected
    /\ \A i \in 1 .. Len(delivered) : delivered[i] = i - 1
====
