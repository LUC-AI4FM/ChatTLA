---- MODULE VectorClock ----
(***************************************************************************)
(* Vector clocks for two processes.  Each event ticks the local entry; a   *)
(* send copies the local vector into the message; on receive the receiver  *)
(* takes elementwise max and ticks its own entry.                          *)
(* Strong safety: each process's own entry monotonically grows; the local  *)
(* clock dominates any vector that it has already received.                *)
(***************************************************************************)
EXTENDS Naturals

Procs == {0, 1}
MaxTick == 3

VARIABLES clock, channel

vars == << clock, channel >>

\* clock[p] is a function 0..1 -> 0..MaxTick
\* channel is a set of vector-clock messages.

ZeroVec == [q \in Procs |-> 0]

Init ==
    /\ clock = [p \in Procs |-> ZeroVec]
    /\ channel = {}

Max(a, b) == IF a >= b THEN a ELSE b

\* Internal event at p: tick p's own entry.
LocalEvent(p) ==
    /\ clock[p][p] < MaxTick
    /\ clock' = [clock EXCEPT ![p] = [clock[p] EXCEPT ![p] = clock[p][p] + 1]]
    /\ UNCHANGED channel

\* Send: tick own entry, then put a copy of local vector on the channel.
Send(p) ==
    /\ clock[p][p] < MaxTick
    /\ LET v0 == [clock[p] EXCEPT ![p] = clock[p][p] + 1] IN
         /\ clock' = [clock EXCEPT ![p] = v0]
         /\ channel' = channel \cup {<<p, v0>>}

\* Receive a message: take elementwise max with its vector and tick own.
Receive(p) ==
    /\ clock[p][p] < MaxTick
    /\ \E m \in channel :
         /\ m[1] # p
         /\ LET merged ==
                  [q \in Procs |->
                     IF q = p
                       THEN Max(clock[p][q], m[2][q]) + 1
                       ELSE Max(clock[p][q], m[2][q])]
            IN  /\ clock' = [clock EXCEPT ![p] = merged]
                /\ channel' = channel \ {m}

Done == UNCHANGED vars

Next ==
    \/ \E p \in Procs : LocalEvent(p)
    \/ \E p \in Procs : Send(p)
    \/ \E p \in Procs : Receive(p)
    \/ Done

Spec == Init /\ [][Next]_vars

\* Strong safety conjoined into TypeOK: own entry never exceeds MaxTick;
\* a clock entry for q seen at p never exceeds q's own entry at q.
TypeOK ==
    /\ clock \in [Procs -> [Procs -> 0 .. MaxTick]]
    /\ channel \subseteq (Procs \X [Procs -> 0 .. MaxTick])
    /\ \A p \in Procs : clock[p][p] <= MaxTick
    /\ \A p, q \in Procs : clock[p][q] <= clock[q][q]
    /\ \A m \in channel : \A q \in Procs : m[2][q] <= clock[q][q]
====
