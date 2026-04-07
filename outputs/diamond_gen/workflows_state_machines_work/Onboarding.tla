---- MODULE Onboarding ----
(***************************************************************************)
(*  User onboarding pipeline. Steps must complete in strict order:         *)
(*    signup -> email_verify -> profile -> terms                           *)
(***************************************************************************)
EXTENDS Naturals

VARIABLES phase, ever_signup, ever_email, ever_profile, ever_terms

vars == << phase, ever_signup, ever_email, ever_profile, ever_terms >>

Phases == {"start", "signed_up", "email_verified", "profile_done", "terms_accepted"}

Init == /\ phase        = "start"
        /\ ever_signup  = FALSE
        /\ ever_email   = FALSE
        /\ ever_profile = FALSE
        /\ ever_terms   = FALSE

Signup == /\ phase = "start"
          /\ phase' = "signed_up"
          /\ ever_signup' = TRUE
          /\ UNCHANGED << ever_email, ever_profile, ever_terms >>

VerifyEmail == /\ phase = "signed_up"
               /\ phase' = "email_verified"
               /\ ever_email' = TRUE
               /\ UNCHANGED << ever_signup, ever_profile, ever_terms >>

CompleteProfile == /\ phase = "email_verified"
                   /\ phase' = "profile_done"
                   /\ ever_profile' = TRUE
                   /\ UNCHANGED << ever_signup, ever_email, ever_terms >>

AcceptTerms == /\ phase = "profile_done"
               /\ phase' = "terms_accepted"
               /\ ever_terms' = TRUE
               /\ UNCHANGED << ever_signup, ever_email, ever_profile >>

Done == /\ phase = "terms_accepted"
        /\ UNCHANGED vars

Next == \/ Signup \/ VerifyEmail \/ CompleteProfile \/ AcceptTerms \/ Done

Spec == Init /\ [][Next]_vars

\* Each phase implies all prior steps completed.
SafetyInvariant == ((phase = "terms_accepted") => (ever_signup /\ ever_email /\ ever_profile /\ ever_terms)) /\ ((phase = "profile_done") => (ever_signup /\ ever_email /\ ever_profile)) /\ ((phase = "email_verified") => (ever_signup /\ ever_email))

TypeOK == /\ phase \in Phases
          /\ ever_signup \in BOOLEAN
          /\ ever_email \in BOOLEAN
          /\ ever_profile \in BOOLEAN
          /\ ever_terms \in BOOLEAN
          /\ SafetyInvariant
====
