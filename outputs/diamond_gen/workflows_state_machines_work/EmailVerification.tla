---- MODULE EmailVerification ----
(***************************************************************************)
(*  Email verification workflow.                                           *)
(*    unverified -> code_sent -> verified                                  *)
(*  Codes are single-use and time-limited.                                 *)
(***************************************************************************)
EXTENDS Naturals

CONSTANT MaxTime

VARIABLES status, code_sent_at, code_used, now, ever_code

vars == << status, code_sent_at, code_used, now, ever_code >>

States == {"unverified", "code_sent", "verified", "expired"}

Init == /\ status      = "unverified"
        /\ code_sent_at= 0
        /\ code_used   = FALSE
        /\ now         = 0
        /\ ever_code   = FALSE

SendCode == /\ status = "unverified"
            /\ status' = "code_sent"
            /\ code_sent_at' = now
            /\ ever_code' = TRUE
            /\ UNCHANGED << code_used, now >>

Verify == /\ status = "code_sent"
          /\ ~code_used
          /\ now <= code_sent_at + 1
          /\ status' = "verified"
          /\ code_used' = TRUE
          /\ UNCHANGED << code_sent_at, now, ever_code >>

Expire == /\ status = "code_sent"
          /\ now > code_sent_at + 1
          /\ status' = "expired"
          /\ UNCHANGED << code_sent_at, code_used, now, ever_code >>

Tick == /\ now < MaxTime
        /\ now' = now + 1
        /\ UNCHANGED << status, code_sent_at, code_used, ever_code >>

Done == /\ \/ status \in {"verified", "expired"} \/ now = MaxTime
        /\ UNCHANGED vars

Next == \/ SendCode \/ Verify \/ Expire \/ Tick \/ Done

Spec == Init /\ [][Next]_vars

\* Verified state implies a single-use code was issued and consumed.
SafetyInvariant == ((status = "verified") => (ever_code /\ code_used)) /\ (code_used => ever_code) /\ ((status = "expired") => ever_code)

TypeOK == /\ status \in States
          /\ code_sent_at \in 0..MaxTime
          /\ code_used \in BOOLEAN
          /\ now \in 0..MaxTime
          /\ ever_code \in BOOLEAN
          /\ SafetyInvariant
====
