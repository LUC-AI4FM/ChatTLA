---- MODULE TokenRing ----
(***************************************************************************)
(* Token-passing ring of N stations.  Exactly one token circulates around  *)
(* the ring; only the token holder may transmit.                           *)
(* Strong safety: the token always exists and is held by exactly one       *)
(* station; only that station can be in "transmitting".                    *)
(***************************************************************************)
EXTENDS Naturals

N == 3
Stations == 0 .. (N - 1)

VARIABLES holder, mode

vars == << holder, mode >>

\* mode[i] in {"idle", "transmitting"}.
Init ==
    /\ holder = 0
    /\ mode = [s \in Stations |-> "idle"]

\* Holder begins transmission.
StartTx(s) ==
    /\ holder = s
    /\ mode[s] = "idle"
    /\ mode' = [mode EXCEPT ![s] = "transmitting"]
    /\ UNCHANGED holder

\* Holder finishes transmitting.
StopTx(s) ==
    /\ holder = s
    /\ mode[s] = "transmitting"
    /\ mode' = [mode EXCEPT ![s] = "idle"]
    /\ UNCHANGED holder

\* Holder passes the token to the next station (must be idle first).
PassToken(s) ==
    /\ holder = s
    /\ mode[s] = "idle"
    /\ holder' = (s + 1) % N
    /\ UNCHANGED mode

Done == UNCHANGED vars

Next ==
    \/ \E s \in Stations : StartTx(s)
    \/ \E s \in Stations : StopTx(s)
    \/ \E s \in Stations : PassToken(s)
    \/ Done

Spec == Init /\ [][Next]_vars

\* Strong safety conjoined into TypeOK: at most one station transmits and
\* the token is the only enabler.
TypeOK ==
    /\ holder \in Stations
    /\ mode \in [Stations -> {"idle", "transmitting"}]
    /\ \A s \in Stations : (mode[s] = "transmitting") => (s = holder)
    /\ \A s, t \in Stations :
         (mode[s] = "transmitting" /\ mode[t] = "transmitting") => (s = t)
====
