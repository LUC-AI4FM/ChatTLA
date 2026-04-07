---- MODULE TwoPhaseLockingDeadlock ----
(***************************************************************************)
(*  Strict 2PL on TWO resources.  We model the classical deadlock that    *)
(*  arises when two transactions take the resources in opposite order.    *)
(*  We track the waits-for graph as a relation over Txns.                 *)
(*                                                                         *)
(*  Strong invariant: a transaction holding both locks is the unique      *)
(*  holder of those locks, and no transaction holds and waits on the same *)
(*  resource simultaneously.                                              *)
(***************************************************************************)
EXTENDS Naturals, FiniteSets

CONSTANTS Txns

Resources == {"R1", "R2"}

VARIABLES holds, waits

vars == << holds, waits >>

\* holds[r] = the unique txn holding r, or NONE.  We use a TLA model value
\* for NONE via the string "none".
NONE == "none"

Init == /\ holds = [r \in Resources |-> NONE]
        /\ waits = [t \in Txns |-> NONE]

\* A txn that is not waiting for anything may try to grab a free lock.
Acquire(t, r) ==
    /\ waits[t] = NONE
    /\ holds[r] = NONE
    /\ holds' = [holds EXCEPT ![r] = t]
    /\ UNCHANGED waits

\* If the lock is held by someone else, the txn enters the waits-for graph.
Wait(t, r) ==
    /\ waits[t] = NONE
    /\ holds[r] # NONE
    /\ holds[r] # t
    /\ waits' = [waits EXCEPT ![t] = holds[r]]
    /\ UNCHANGED holds

\* A txn releases all the resources it holds (commit/abort step).
Release(t) ==
    /\ \E r \in Resources : holds[r] = t
    /\ holds' = [r \in Resources |-> IF holds[r] = t THEN NONE ELSE holds[r]]
    /\ waits' = [u \in Txns |-> IF waits[u] = t THEN NONE ELSE waits[u]]

Next == \/ \E t \in Txns, r \in Resources : Acquire(t,r)
        \/ \E t \in Txns, r \in Resources : Wait(t,r)
        \/ \E t \in Txns : Release(t)

Spec == Init /\ [][Next]_vars /\ WF_vars(Next)

\* Strong invariant: each lock has at most one holder, and a waiting
\* transaction is waiting for the actual holder.  Conjoined into TypeOK.
TypeOK == /\ holds \in [Resources -> Txns \cup {NONE}]
          /\ waits \in [Txns -> Txns \cup {NONE}]
          /\ \A t \in Txns :
                waits[t] # NONE => \E r \in Resources : holds[r] = waits[t]
          /\ \A t \in Txns : waits[t] # t
====
