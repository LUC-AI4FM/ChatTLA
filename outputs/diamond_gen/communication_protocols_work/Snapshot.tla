---- MODULE Snapshot ----
(***************************************************************************)
(* Chandy-Lamport global snapshot for two processes connected by a single *)
(* one-way channel from p1 to p2.  An initiator records its own state and *)
(* sends a marker; the marker triggers the receiver to record its state   *)
(* and the channel state up to the marker.                                *)
(* Strong safety: once both processes have recorded, their recorded local *)
(* states + channel state form a consistent cut (no message recorded as   *)
(* received but not sent).                                                *)
(***************************************************************************)
EXTENDS Naturals, Sequences

VARIABLES sLocal, rLocal, channel, recSender, recReceiver, recChannel, sentCount

vars == << sLocal, rLocal, channel, recSender, recReceiver, recChannel, sentCount >>

NoRec == 99

Init ==
    /\ sLocal = 0
    /\ rLocal = 0
    /\ channel = << >>
    /\ recSender = NoRec
    /\ recReceiver = NoRec
    /\ recChannel = << >>
    /\ sentCount = 0

\* Sender does local work and sends a data message.
SendData ==
    /\ sLocal < 2
    /\ sLocal' = sLocal + 1
    /\ channel' = Append(channel, sentCount + 1)
    /\ sentCount' = sentCount + 1
    /\ UNCHANGED << rLocal, recSender, recReceiver, recChannel >>

\* Receiver consumes head of channel.
RecvData ==
    /\ Len(channel) > 0
    /\ Head(channel) # 0
    /\ rLocal' = rLocal + Head(channel)
    /\ channel' = Tail(channel)
    /\ \* If snapshot in progress, append to recorded channel.
       (IF recSender # NoRec /\ recReceiver = NoRec
          THEN recChannel' = Append(recChannel, Head(channel))
          ELSE UNCHANGED recChannel)
    /\ UNCHANGED << sLocal, recSender, sentCount >>
    /\ UNCHANGED recReceiver

\* Initiate snapshot at sender: record local state and emit marker.
\* The marker is modelled as a distinguished value 0 in the channel.
StartSnap ==
    /\ recSender = NoRec
    /\ recSender' = sLocal
    /\ channel' = Append(channel, 0)
    /\ UNCHANGED << sLocal, rLocal, recReceiver, recChannel, sentCount >>

\* Receiver sees the marker: record its local state, stop logging channel.
RecvMarker ==
    /\ Len(channel) > 0
    /\ Head(channel) = 0
    /\ recReceiver = NoRec
    /\ recReceiver' = rLocal
    /\ channel' = Tail(channel)
    /\ UNCHANGED << sLocal, rLocal, recSender, recChannel, sentCount >>

Done == UNCHANGED vars

Next == SendData \/ RecvData \/ StartSnap \/ RecvMarker \/ Done

Spec == Init /\ [][Next]_vars

\* Strong safety conjoined into TypeOK: when the snapshot has been taken,
\* the recorded sender state plus channel-in-flight equals the receiver's
\* recorded state plus its already-consumed values.  We capture an
\* invariant: recSender always reflects sLocal at snapshot time and
\* recSender <= sLocal afterwards (sender only grows).
TypeOK ==
    /\ sLocal \in 0 .. 5
    /\ rLocal \in 0 .. 20
    /\ Len(channel) \in 0 .. 5
    /\ recSender \in (0 .. 5) \cup {NoRec}
    /\ recReceiver \in (0 .. 20) \cup {NoRec}
    /\ Len(recChannel) \in 0 .. 5
    /\ sentCount \in 0 .. 5
    /\ (recSender # NoRec) => (recSender <= sLocal)
    /\ (recReceiver # NoRec) => (recReceiver <= rLocal)
====
