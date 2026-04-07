---- MODULE VersionedKv ----
EXTENDS Naturals

CONSTANTS Keys, MaxVersion

\* Bounded versioned KV: each Put bumps a global version; the latest write per key wins.
VARIABLES versions, store

vars == << versions, store >>

Init == /\ versions = [k \in Keys |-> 0]
        /\ store = [k \in Keys |-> 0]

\* Put a fresh value (modeled as the next version number itself) for key k.
Put(k) == /\ k \in Keys
          /\ versions[k] < MaxVersion
          /\ versions' = [versions EXCEPT ![k] = @ + 1]
          /\ store' = [store EXCEPT ![k] = versions[k] + 1]

\* Get is a no-op observation.
Get(k) == /\ k \in Keys
          /\ UNCHANGED vars

Next == (\E k \in Keys : Put(k)) \/ (\E k \in Keys : Get(k))

Spec == Init /\ [][Next]_vars /\ WF_vars(Next)

\* Strong invariant: stored value equals current version (max put so far).
Valid == /\ \A k \in Keys : versions[k] \in 0..MaxVersion
         /\ \A k \in Keys : store[k] = versions[k]

TypeOK == /\ versions \in [Keys -> 0..MaxVersion]
          /\ store \in [Keys -> 0..MaxVersion]
          /\ Valid
====
