---- MODULE LamportClock ----
(***************************************************************************)
(* Lamport scalar logical clocks for two processes.                        *)
(* Each event increments the local clock; on receive the local clock is    *)
(* set to max(local, sent) + 1.  All event timestamps are uniquely ordered *)
(* by the (clock, pid) tuple.                                              *)
(* Strong safety: clocks are bounded above by MaxClock and a sent message  *)
(* timestamp never exceeds the sender's current clock.                     *)
(***************************************************************************)
EXTENDS Naturals

Procs == {0, 1}
MaxClock == 4

VARIABLES clock, channel

vars == << clock, channel >>

\* channel = set of <<sender, timestamp>> messages.
Init ==
    /\ clock = [p \in Procs |-> 0]
    /\ channel = {}

Max(a, b) == IF a >= b THEN a ELSE b

Local(p) ==
    /\ clock[p] < MaxClock
    /\ clock' = [clock EXCEPT ![p] = clock[p] + 1]
    /\ UNCHANGED channel

Send(p) ==
    /\ clock[p] < MaxClock
    /\ clock' = [clock EXCEPT ![p] = clock[p] + 1]
    /\ channel' = channel \cup {<<p, clock[p] + 1>>}

Receive(p) ==
    /\ \E m \in channel :
         /\ m[1] # p
         /\ Max(clock[p], m[2]) + 1 <= MaxClock
         /\ clock' = [clock EXCEPT ![p] = Max(clock[p], m[2]) + 1]
         /\ channel' = channel \ {m}

Done == UNCHANGED vars

Next ==
    \/ \E p \in Procs : Local(p)
    \/ \E p \in Procs : Send(p)
    \/ \E p \in Procs : Receive(p)
    \/ Done

Spec == Init /\ [][Next]_vars

\* Strong safety conjoined into TypeOK: clocks bounded; any in-flight
\* message timestamp is <= sender's current clock.
TypeOK ==
    /\ clock \in [Procs -> 0 .. MaxClock]
    /\ channel \subseteq (Procs \X (1 .. MaxClock))
    /\ \A m \in channel : m[2] <= clock[m[1]]
====
