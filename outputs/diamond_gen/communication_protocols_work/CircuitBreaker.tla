---- MODULE CircuitBreaker ----
(***************************************************************************)
(* Circuit-breaker pattern with three cbStates: closed, open, half-open.    *)
(*  - In CLOSED, requests pass; consecutive failures trip the breaker.    *)
(*  - In OPEN, requests fail fast; a timeout transitions to HALF_OPEN.    *)
(*  - In HALF_OPEN, exactly one probe is admitted; success closes, fail   *)
(*    re-opens.                                                            *)
(* Strong safety: HALF_OPEN admits at most one probe at a time, and the   *)
(* failure counter only matters in CLOSED.                                 *)
(***************************************************************************)
EXTENDS Naturals

FailLimit == 2

VARIABLES cbState, failures, probeInFlight

vars == << cbState, failures, probeInFlight >>

States == {"closed", "open", "half_open"}

Init ==
    /\ cbState = "closed"
    /\ failures = 0
    /\ probeInFlight = FALSE

\* CLOSED: a successful request resets the failure counter.
ClosedSuccess ==
    /\ cbState = "closed"
    /\ failures' = 0
    /\ UNCHANGED << cbState, probeInFlight >>

\* CLOSED: a failed request bumps the failure counter; if it crosses the
\* threshold the breaker trips OPEN.
ClosedFailure ==
    /\ cbState = "closed"
    /\ failures < FailLimit
    /\ LET f1 == failures + 1 IN
         /\ cbState' = IF f1 = FailLimit THEN "open" ELSE "closed"
         /\ failures' = IF f1 = FailLimit THEN 0 ELSE f1
    /\ UNCHANGED probeInFlight

\* OPEN: timeout fires, transition to HALF_OPEN.
TimeoutToHalfOpen ==
    /\ cbState = "open"
    /\ cbState' = "half_open"
    /\ failures' = 0
    /\ UNCHANGED probeInFlight

\* HALF_OPEN: admit a single probe (only when none is in flight).
StartProbe ==
    /\ cbState = "half_open"
    /\ probeInFlight = FALSE
    /\ probeInFlight' = TRUE
    /\ UNCHANGED << cbState, failures >>

\* Probe succeeds: close the breaker.
ProbeSuccess ==
    /\ cbState = "half_open"
    /\ probeInFlight = TRUE
    /\ probeInFlight' = FALSE
    /\ cbState' = "closed"
    /\ failures' = 0

\* Probe fails: re-open.
ProbeFailure ==
    /\ cbState = "half_open"
    /\ probeInFlight = TRUE
    /\ probeInFlight' = FALSE
    /\ cbState' = "open"
    /\ failures' = 0

Done == UNCHANGED vars

Next ==
    \/ ClosedSuccess \/ ClosedFailure \/ TimeoutToHalfOpen
    \/ StartProbe \/ ProbeSuccess \/ ProbeFailure \/ Done

Spec == Init /\ [][Next]_vars

\* Strong safety conjoined into TypeOK.
TypeOK ==
    /\ cbState \in States
    /\ failures \in 0 .. FailLimit
    /\ probeInFlight \in BOOLEAN
    \* Probe is only meaningful in half-open.
    /\ (probeInFlight = TRUE) => (cbState = "half_open")
    \* failures = threshold only ever observed in CLOSED transiently is
    \* not allowed: in CLOSED failures < threshold.
    /\ (cbState = "closed") => (failures < FailLimit)
    /\ (cbState = "open") => (failures = 0)
    /\ (cbState = "half_open") => (failures = 0)
====
