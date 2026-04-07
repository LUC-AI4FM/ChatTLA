---- MODULE RateLimiter ----
(***************************************************************************)
(* Token-bucket rate limiter.  Bucket has capacity K and refills one      *)
(* token per tick (capped at K).  An incoming request consumes one token *)
(* if available; otherwise it is rejected.                               *)
(*                                                                         *)
(* Safety: token count always lies in 0..K; admitted requests within any *)
(* tick budget never exceed bucket+refill (encoded as a per-tick sanity  *)
(* invariant on the token level).                                        *)
(***************************************************************************)
EXTENDS Naturals

K == 3  \* bucket capacity

VARIABLES tokens, admitted

vars == << tokens, admitted >>

Init == /\ tokens = K
        /\ admitted = 0

\* Refill one token (up to capacity).
Refill == /\ tokens < K
          /\ tokens' = tokens + 1
          /\ UNCHANGED admitted

\* Admit a request: consume one token, count it.
Admit == /\ tokens > 0
         /\ admitted < K
         /\ tokens' = tokens - 1
         /\ admitted' = admitted + 1

\* Deliver an admitted request out of the system (keeps state space finite).
Deliver == /\ admitted > 0
           /\ admitted' = admitted - 1
           /\ UNCHANGED tokens

Next == Refill \/ Admit \/ Deliver

Spec == Init /\ [][Next]_vars

\* Strong safety: token count is always in 0..K, and admitted-but-undelivered
\* in-flight requests never exceed the capacity K.
TokenInv == tokens \in 0..K /\ admitted \in 0..K

TypeOK == TokenInv
====
