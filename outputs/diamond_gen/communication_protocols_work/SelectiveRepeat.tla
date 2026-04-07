---- MODULE SelectiveRepeat ----
(***************************************************************************)
(* Selective-repeat ARQ.  Sender keeps an unacked window [base, next) of   *)
(* size <= W and a per-seq ack flag.  Receiver buffers frames in window    *)
(* [rbase, rbase + W) and delivers contiguous prefix.                      *)
(* Strong safety: in-order delivery to the upper layer with no gaps.       *)
(***************************************************************************)
EXTENDS Naturals, Sequences

W == 2
MaxSeq == 4

VARIABLES base, next, acked, rbase, buffer, channel, delivered

vars == << base, next, acked, rbase, buffer, channel, delivered >>

Init ==
    /\ base = 0
    /\ next = 0
    /\ acked = [s \in 0 .. (MaxSeq - 1) |-> FALSE]
    /\ rbase = 0
    /\ buffer = [s \in 0 .. (MaxSeq - 1) |-> FALSE]
    /\ channel = {}
    /\ delivered = << >>

\* Sender transmits a fresh frame.
SendNew ==
    /\ next < MaxSeq
    /\ next - base < W
    /\ channel' = channel \cup {next}
    /\ next' = next + 1
    /\ UNCHANGED << base, acked, rbase, buffer, delivered >>

\* Sender selectively retransmits an unacked frame in window.
Retransmit ==
    /\ \E s \in 0 .. (MaxSeq - 1) :
         /\ base <= s /\ s < next
         /\ acked[s] = FALSE
         /\ channel' = channel \cup {s}
    /\ UNCHANGED << base, next, acked, rbase, buffer, delivered >>

DropFrame ==
    /\ \E s \in channel : channel' = channel \ {s}
    /\ UNCHANGED << base, next, acked, rbase, buffer, delivered >>

\* Receiver buffers any in-window frame.
RecvFrame ==
    /\ \E s \in channel :
         /\ rbase <= s /\ s < rbase + W
         /\ buffer' = [buffer EXCEPT ![s] = TRUE]
         /\ channel' = channel \ {s}
         /\ acked' = [acked EXCEPT ![s] = TRUE]
    /\ UNCHANGED << base, next, rbase, delivered >>

\* Receiver delivers the contiguous prefix from rbase.
DeliverNext ==
    /\ rbase < MaxSeq
    /\ buffer[rbase] = TRUE
    /\ delivered' = Append(delivered, rbase)
    /\ buffer' = [buffer EXCEPT ![rbase] = FALSE]
    /\ rbase' = rbase + 1
    /\ UNCHANGED << base, next, acked, channel >>

\* Sender slides base over fully acked prefix.
SlideBase ==
    /\ base < next
    /\ acked[base] = TRUE
    /\ base' = base + 1
    /\ UNCHANGED << next, acked, rbase, buffer, channel, delivered >>

Done ==
    /\ base = MaxSeq
    /\ rbase = MaxSeq
    /\ UNCHANGED vars

Next ==
    \/ SendNew \/ Retransmit \/ DropFrame
    \/ RecvFrame \/ DeliverNext \/ SlideBase \/ Done

Spec == Init /\ [][Next]_vars

TypeOK ==
    /\ base \in 0 .. MaxSeq
    /\ next \in 0 .. MaxSeq
    /\ rbase \in 0 .. MaxSeq
    /\ acked \in [0 .. (MaxSeq - 1) -> BOOLEAN]
    /\ buffer \in [0 .. (MaxSeq - 1) -> BOOLEAN]
    /\ channel \subseteq 0 .. (MaxSeq - 1)
    /\ Len(delivered) \in 0 .. MaxSeq
    /\ base <= next
    /\ next - base <= W
    /\ Len(delivered) = rbase
    /\ \A i \in 1 .. Len(delivered) : delivered[i] = i - 1
====
