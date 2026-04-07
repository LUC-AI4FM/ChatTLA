---- MODULE OAuth2Flow ----
(***************************************************************************)
(*  OAuth 2.0 authorization-code flow.                                     *)
(*    start -> consented -> code_issued -> token_issued -> resource_access *)
(*  Resource access requires a token, which requires a code, which         *)
(*  requires explicit user consent. Denial branch: start -> denied.        *)
(***************************************************************************)
EXTENDS Naturals

VARIABLES phase, ever_consented, ever_code, ever_token

vars == << phase, ever_consented, ever_code, ever_token >>

States == {"start", "consented", "code_issued", "token_issued", "resource_access", "denied"}

Init == /\ phase           = "start"
        /\ ever_consented   = FALSE
        /\ ever_code        = FALSE
        /\ ever_token       = FALSE

Consent == /\ phase = "start"
           /\ phase' = "consented"
           /\ ever_consented' = TRUE
           /\ UNCHANGED << ever_code, ever_token >>

Deny == /\ phase = "start"
        /\ phase' = "denied"
        /\ UNCHANGED << ever_consented, ever_code, ever_token >>

IssueCode == /\ phase = "consented"
             /\ phase' = "code_issued"
             /\ ever_code' = TRUE
             /\ UNCHANGED << ever_consented, ever_token >>

IssueToken == /\ phase = "code_issued"
              /\ phase' = "token_issued"
              /\ ever_token' = TRUE
              /\ UNCHANGED << ever_consented, ever_code >>

AccessResource == /\ phase = "token_issued"
                  /\ phase' = "resource_access"
                  /\ UNCHANGED << ever_consented, ever_code, ever_token >>

Done == /\ phase \in {"resource_access", "denied"}
        /\ UNCHANGED vars

Next == \/ Consent \/ Deny \/ IssueCode \/ IssueToken \/ AccessResource \/ Done

Spec == Init /\ [][Next]_vars

\* Resource access implies a token, which implies an authorization code,
\* which implies the user explicitly consented at some prior step.
SafetyInvariant == ((phase = "resource_access") => (ever_token /\ ever_code /\ ever_consented)) /\ ((phase = "denied") => (~ever_token))

TypeOK == /\ phase \in States
          /\ ever_consented \in BOOLEAN
          /\ ever_code      \in BOOLEAN
          /\ ever_token     \in BOOLEAN
          /\ SafetyInvariant
====
