---- MODULE KvStore ----
EXTENDS Naturals

CONSTANTS Keys, Vals

NONE == "NONE"

VARIABLES store  \* function Keys -> Vals \cup {NONE}

vars == << store >>

Init == store = [k \in Keys |-> NONE]

Put(k, v) == /\ k \in Keys
             /\ v \in Vals
             /\ store' = [store EXCEPT ![k] = v]

Delete(k) == /\ k \in Keys
             /\ store[k] # NONE
             /\ store' = [store EXCEPT ![k] = NONE]

\* Get is an observation; state unchanged.
Get(k) == /\ k \in Keys
          /\ UNCHANGED vars

Next == (\E k \in Keys, v \in Vals : Put(k, v))
        \/ (\E k \in Keys : Delete(k))
        \/ (\E k \in Keys : Get(k))

Spec == Init /\ [][Next]_vars

\* Strong invariant: every key maps to NONE or a valid value.
Valid == \A k \in Keys : store[k] \in Vals \cup {NONE}

TypeOK == /\ store \in [Keys -> Vals \cup {NONE}]
          /\ Valid
====
