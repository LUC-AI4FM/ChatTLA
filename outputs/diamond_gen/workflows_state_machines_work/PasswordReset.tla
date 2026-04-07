---- MODULE PasswordReset ----
(***************************************************************************)
(*  Password reset workflow.                                               *)
(*    idle -> token_issued -> token_used (success) | token_expired         *)
(*  Tokens are single-use and time-bounded.                                *)
(***************************************************************************)
EXTENDS Naturals

CONSTANT MaxTime

VARIABLES status, issued_at, now, token_used, ever_issued

vars == << status, issued_at, now, token_used, ever_issued >>

States == {"idle", "token_issued", "reset_done", "expired"}

Init == /\ status      = "idle"
        /\ issued_at   = 0
        /\ now         = 0
        /\ token_used  = FALSE
        /\ ever_issued = FALSE

Issue == /\ status = "idle"
         /\ status' = "token_issued"
         /\ issued_at' = now
         /\ ever_issued' = TRUE
         /\ UNCHANGED << now, token_used >>

UseToken == /\ status = "token_issued"
            /\ ~token_used
            /\ now <= issued_at + 1
            /\ status' = "reset_done"
            /\ token_used' = TRUE
            /\ UNCHANGED << issued_at, now, ever_issued >>

Expire == /\ status = "token_issued"
          /\ now > issued_at + 1
          /\ status' = "expired"
          /\ UNCHANGED << issued_at, now, token_used, ever_issued >>

Tick == /\ now < MaxTime
        /\ now' = now + 1
        /\ UNCHANGED << status, issued_at, token_used, ever_issued >>

Done == /\ \/ status \in {"reset_done", "expired"} \/ now = MaxTime
        /\ UNCHANGED vars

Next == \/ Issue \/ UseToken \/ Expire \/ Tick \/ Done

Spec == Init /\ [][Next]_vars

\* A successful reset consumed an unexpired, previously issued token.
SafetyInvariant == ((status = "reset_done") => (ever_issued /\ token_used)) /\ (token_used => ever_issued) /\ ((status = "expired") => ever_issued)

TypeOK == /\ status \in States
          /\ issued_at \in 0..MaxTime
          /\ now \in 0..MaxTime
          /\ token_used \in BOOLEAN
          /\ ever_issued \in BOOLEAN
          /\ SafetyInvariant
====
