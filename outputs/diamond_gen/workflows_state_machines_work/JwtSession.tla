---- MODULE JwtSession ----
(***************************************************************************)
(*  Stateful model of a JWT session lifecycle.                             *)
(*  A token is issued, optionally refreshed, then either expires or is     *)
(*  revoked. Acceptance requires the token to be alive (issued and not     *)
(*  expired/revoked). The current_time is bounded so the state space is    *)
(*  finite.                                                                *)
(***************************************************************************)
EXTENDS Naturals

CONSTANT MaxTime

VARIABLES status, expiry, now, ever_issued, ever_revoked

vars == << status, expiry, now, ever_issued, ever_revoked >>

States == {"none", "active", "expired", "revoked"}

Init == /\ status      = "none"
        /\ expiry      = 0
        /\ now         = 0
        /\ ever_issued = FALSE
        /\ ever_revoked= FALSE

Issue == /\ status = "none"
         /\ status' = "active"
         /\ expiry' = now + 2
         /\ ever_issued' = TRUE
         /\ UNCHANGED << now, ever_revoked >>

Refresh == /\ status = "active"
           /\ now < expiry
           /\ expiry' = now + 2
           /\ UNCHANGED << status, now, ever_issued, ever_revoked >>

Tick == /\ now < MaxTime
        /\ now' = now + 1
        /\ status' = IF status = "active" /\ now + 1 >= expiry THEN "expired" ELSE status
        /\ UNCHANGED << expiry, ever_issued, ever_revoked >>

Revoke == /\ status = "active"
          /\ status' = "revoked"
          /\ ever_revoked' = TRUE
          /\ UNCHANGED << expiry, now, ever_issued >>

Done == /\ \/ status \in {"expired", "revoked"}
           \/ now = MaxTime
        /\ UNCHANGED vars

Next == \/ Issue \/ Refresh \/ Tick \/ Revoke \/ Done

Spec == Init /\ [][Next]_vars

\* Any active session was issued and has not expired and has not been revoked.
SafetyInvariant == ((status = "active") => (ever_issued /\ now < expiry /\ ~ever_revoked)) /\ ((status = "revoked") => ever_issued)

TypeOK == /\ status \in States
          /\ expiry \in 0..(MaxTime + 5)
          /\ now    \in 0..MaxTime
          /\ ever_issued  \in BOOLEAN
          /\ ever_revoked \in BOOLEAN
          /\ SafetyInvariant
====
