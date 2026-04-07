---- MODULE IdempotencyKey ----
(***************************************************************************)
(*  Idempotency key.  Each client request carries a key.  The server     *)
(*  stores key -> result; on retry, the server returns the cached       *)
(*  result instead of re-executing the side-effect.                     *)
(*                                                                         *)
(*  Strong invariant: for every key, all observed results agree, and    *)
(*  the side-effect counter equals the number of distinct keys served.  *)
(***************************************************************************)
EXTENDS Naturals, FiniteSets

CONSTANTS Keys

NONE == 0

VARIABLES cache, sideEffects, observed

vars == << cache, sideEffects, observed >>

Init == /\ cache       = [k \in Keys |-> NONE]
        /\ sideEffects = 0
        /\ observed    = [k \in Keys |-> NONE]

\* First time we see this key: run the side-effect, generate a result.
FirstCall(k) == /\ cache[k] = NONE
                /\ cache'       = [cache       EXCEPT ![k] = sideEffects + 1]
                /\ sideEffects' = sideEffects + 1
                /\ observed'    = [observed    EXCEPT ![k] = sideEffects + 1]

\* Retry: same key, return cached result, no new side effect.
Retry(k) == /\ cache[k] # NONE
            /\ observed' = [observed EXCEPT ![k] = cache[k]]
            /\ UNCHANGED << cache, sideEffects >>

Next == \/ \E k \in Keys : FirstCall(k)
        \/ \E k \in Keys : Retry(k)

Spec == Init /\ [][Next]_vars /\ WF_vars(Next)

\* Strong invariant: observed[k] is always the cached result, side-effect
\* count = number of cached keys.
ServedKeys == {k \in Keys : cache[k] # NONE}

TypeOK == /\ cache       \in [Keys -> 0..Cardinality(Keys)]
          /\ sideEffects \in 0..Cardinality(Keys)
          /\ observed    \in [Keys -> 0..Cardinality(Keys)]
          /\ sideEffects = Cardinality(ServedKeys)
          /\ \A k \in Keys :
                observed[k] # NONE => observed[k] = cache[k]
====
