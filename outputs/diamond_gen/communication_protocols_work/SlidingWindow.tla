---- MODULE SlidingWindow ----
(***************************************************************************)
(* Sliding-window protocol over a reliable, in-order channel.              *)
(* Sender keeps a base (oldest unacked) and next (next to send), with at   *)
(* most W frames in flight.  Receiver delivers in order from "expected".   *)
(* Strong safety: in-order delivery and gap = next - base never exceeds W. *)
(***************************************************************************)
EXTENDS Naturals, Sequences

W == 2
MaxSeq == 4

VARIABLES base, next, expected, channel, delivered

vars == << base, next, expected, channel, delivered >>

Init ==
    /\ base = 0
    /\ next = 0
    /\ expected = 0
    /\ channel = << >>
    /\ delivered = << >>

\* Sender pushes seq number "next" if window has room.
SendFrame ==
    /\ next < MaxSeq
    /\ next - base < W
    /\ channel' = Append(channel, next)
    /\ next' = next + 1
    /\ UNCHANGED << base, expected, delivered >>

\* Receiver delivers the head if it matches "expected".
DeliverHead ==
    /\ Len(channel) > 0
    /\ Head(channel) = expected
    /\ delivered' = Append(delivered, expected)
    /\ expected' = expected + 1
    /\ channel' = Tail(channel)
    /\ UNCHANGED << base, next >>

\* Sender slides base when receiver has progressed.
AdvanceBase ==
    /\ base < expected
    /\ base' = base + 1
    /\ UNCHANGED << next, expected, channel, delivered >>

Done ==
    /\ base = MaxSeq
    /\ next = MaxSeq
    /\ expected = MaxSeq
    /\ channel = << >>
    /\ UNCHANGED vars

Next == SendFrame \/ DeliverHead \/ AdvanceBase \/ Done

Spec == Init /\ [][Next]_vars

\* Strong safety conjoined into TypeOK: in-order prefix delivery and the
\* sliding window invariant base <= expected <= next, next - base <= W.
TypeOK ==
    /\ base \in 0 .. MaxSeq
    /\ next \in 0 .. MaxSeq
    /\ expected \in 0 .. MaxSeq
    /\ Len(channel) \in 0 .. W
    /\ Len(delivered) \in 0 .. MaxSeq
    /\ base <= expected
    /\ expected <= next
    /\ next - base <= W
    /\ Len(delivered) = expected
    /\ \A i \in 1 .. Len(delivered) : delivered[i] = i - 1
====
