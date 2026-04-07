---- MODULE TwoFactorAuth ----
(***************************************************************************)
(*  TOTP-style two-factor authentication.                                  *)
(*    none -> factor1_ok -> factor2_ok -> session_granted                  *)
(*    Wrong factor at any step -> denied.                                  *)
(*  Session granted iff both factors verified along the path.              *)
(***************************************************************************)
EXTENDS Naturals

VARIABLES phase, ever_factor1, ever_factor2

vars == << phase, ever_factor1, ever_factor2 >>

States == {"none", "factor1_ok", "factor2_ok", "granted", "denied"}

Init == /\ phase        = "none"
        /\ ever_factor1 = FALSE
        /\ ever_factor2 = FALSE

VerifyFactor1 == /\ phase = "none"
                 /\ phase' = "factor1_ok"
                 /\ ever_factor1' = TRUE
                 /\ UNCHANGED ever_factor2

FailFactor1 == /\ phase = "none"
               /\ phase' = "denied"
               /\ UNCHANGED << ever_factor1, ever_factor2 >>

VerifyFactor2 == /\ phase = "factor1_ok"
                 /\ phase' = "factor2_ok"
                 /\ ever_factor2' = TRUE
                 /\ UNCHANGED ever_factor1

FailFactor2 == /\ phase = "factor1_ok"
               /\ phase' = "denied"
               /\ UNCHANGED << ever_factor1, ever_factor2 >>

GrantSession == /\ phase = "factor2_ok"
                /\ phase' = "granted"
                /\ UNCHANGED << ever_factor1, ever_factor2 >>

Done == /\ phase \in {"granted", "denied"}
        /\ UNCHANGED vars

Next == \/ VerifyFactor1 \/ FailFactor1 \/ VerifyFactor2 \/ FailFactor2 \/ GrantSession \/ Done

Spec == Init /\ [][Next]_vars

\* A granted session implies both factors were independently verified.
SafetyInvariant == ((phase = "granted") => (ever_factor1 /\ ever_factor2)) /\ ((phase = "factor2_ok") => ever_factor1)

TypeOK == /\ phase \in States
          /\ ever_factor1 \in BOOLEAN
          /\ ever_factor2 \in BOOLEAN
          /\ SafetyInvariant
====
