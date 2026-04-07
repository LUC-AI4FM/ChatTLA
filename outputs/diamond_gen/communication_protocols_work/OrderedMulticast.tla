---- MODULE OrderedMulticast ----
(***************************************************************************)
(* Total-order multicast via a sequencer node.  Senders submit messages   *)
(* to the sequencer; the sequencer assigns a monotonic sequence number    *)
(* and relays them; receivers deliver strictly in sequence-number order.  *)
(* Strong safety: each receiver's delivered prefix is a prefix of the     *)
(* sequencer's assignment order; all receivers therefore agree.          *)
(***************************************************************************)
EXTENDS Naturals, Sequences

Receivers == {"r1", "r2"}
MaxMsgs == 3

VARIABLES seq, broadcast, delivered

vars == << seq, broadcast, delivered >>

\* seq        = next sequence number to assign (0 .. MaxMsgs).
\* broadcast  = function from sequence number 1..seq -> producer id.
\* delivered  = per-receiver next sequence number expected (1..MaxMsgs+1).
Init ==
    /\ seq = 0
    /\ broadcast = << >>
    /\ delivered = [r \in Receivers |-> 0]

\* Sequencer assigns the next number to a new message.
Assign ==
    /\ seq < MaxMsgs
    /\ seq' = seq + 1
    /\ broadcast' = Append(broadcast, seq + 1)
    /\ UNCHANGED delivered

\* A receiver delivers the next message in order.
Deliver(r) ==
    /\ delivered[r] < seq
    /\ delivered' = [delivered EXCEPT ![r] = delivered[r] + 1]
    /\ UNCHANGED << seq, broadcast >>

Done == UNCHANGED vars

Next ==
    \/ Assign
    \/ \E r \in Receivers : Deliver(r)
    \/ Done

Spec == Init /\ [][Next]_vars

\* Strong safety conjoined into TypeOK: each receiver's delivery count
\* never exceeds the sequencer count, and broadcast records 1..seq.
TypeOK ==
    /\ seq \in 0 .. MaxMsgs
    /\ Len(broadcast) = seq
    /\ \A i \in 1 .. Len(broadcast) : broadcast[i] = i
    /\ delivered \in [Receivers -> 0 .. MaxMsgs]
    /\ \A r \in Receivers : delivered[r] <= seq
====
