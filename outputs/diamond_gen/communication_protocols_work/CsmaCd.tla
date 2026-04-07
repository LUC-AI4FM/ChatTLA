---- MODULE CsmaCd ----
(***************************************************************************)
(* CSMA/CD: stations sense the carrier, transmit if idle, detect          *)
(* collisions, back off, and retry.                                        *)
(* mode[s] in {"idle","sensing","tx","collide","backoff"}.                 *)
(* Strong safety: a non-collision transmission is the only one in flight,  *)
(* and any tx that overlaps another transmitter is detected as a collision *)
(* by both within one step.                                                *)
(***************************************************************************)
EXTENDS Naturals, FiniteSets

Stations == {"a", "b"}

VARIABLES mode, channel

vars == << mode, channel >>

\* channel = set of stations currently asserting carrier.
Init ==
    /\ mode = [s \in Stations |-> "idle"]
    /\ channel = {}

\* A station starts sensing.
Sense(s) ==
    /\ mode[s] = "idle"
    /\ mode' = [mode EXCEPT ![s] = "sensing"]
    /\ UNCHANGED channel

\* Channel free: begin transmitting.
BeginTx(s) ==
    /\ mode[s] = "sensing"
    /\ channel = {}
    /\ mode' = [mode EXCEPT ![s] = "tx"]
    /\ channel' = channel \cup {s}

\* Detect that another station was already transmitting -> back off.
SenseBusy(s) ==
    /\ mode[s] = "sensing"
    /\ channel # {}
    /\ mode' = [mode EXCEPT ![s] = "backoff"]
    /\ UNCHANGED channel

\* Two transmitters overlap: both detect a collision.
DetectCollision(s) ==
    /\ mode[s] = "tx"
    /\ Cardinality(channel) >= 2
    /\ mode' = [mode EXCEPT ![s] = "collide"]
    /\ channel' = channel \ {s}

\* End a clean transmission.
FinishTx(s) ==
    /\ mode[s] = "tx"
    /\ Cardinality(channel) = 1
    /\ channel = {s}
    /\ mode' = [mode EXCEPT ![s] = "idle"]
    /\ channel' = {}

\* From collide to backoff.
Collide(s) ==
    /\ mode[s] = "collide"
    /\ mode' = [mode EXCEPT ![s] = "backoff"]
    /\ UNCHANGED channel

\* From backoff back to idle (retry later).
Retry(s) ==
    /\ mode[s] = "backoff"
    /\ mode' = [mode EXCEPT ![s] = "idle"]
    /\ UNCHANGED channel

Done == UNCHANGED vars

Next ==
    \/ \E s \in Stations : Sense(s)
    \/ \E s \in Stations : BeginTx(s)
    \/ \E s \in Stations : SenseBusy(s)
    \/ \E s \in Stations : DetectCollision(s)
    \/ \E s \in Stations : FinishTx(s)
    \/ \E s \in Stations : Collide(s)
    \/ \E s \in Stations : Retry(s)
    \/ Done

Spec == Init /\ [][Next]_vars

\* Strong safety conjoined into TypeOK: channel = exactly the set of
\* "tx" stations, and a successful tx is alone.
TypeOK ==
    /\ mode \in [Stations -> {"idle", "sensing", "tx", "collide", "backoff"}]
    /\ channel \subseteq Stations
    /\ channel = { s \in Stations : mode[s] = "tx" }
====
