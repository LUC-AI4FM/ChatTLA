---- MODULE SerializableSnapshot ----
(***************************************************************************)
(*  Serializable snapshot isolation (SSI).  We track read-write antide-  *)
(*  pendencies between concurrent transactions and abort one when an    *)
(*  rw-cycle would form.                                                 *)
(*                                                                         *)
(*  Strong invariant: no committed transaction is the target AND the    *)
(*  source of a rw-antidependency that creates a 2-cycle.               *)
(***************************************************************************)
EXTENDS Naturals, FiniteSets

CONSTANTS Txns

VARIABLES txState, rwOut, rwIn

vars == << txState, rwOut, rwIn >>

States == {"running", "committed", "aborted"}

Init == /\ txState = [t \in Txns |-> "running"]
        /\ rwOut   = [t \in Txns |-> {}]
        /\ rwIn    = [t \in Txns |-> {}]

\* Record an rw-antidependency from t1 to t2 (t1 read what t2 then wrote).
RWConflict(t1, t2) ==
    /\ t1 # t2
    /\ txState[t1] = "running"
    /\ txState[t2] = "running"
    /\ rwOut' = [rwOut EXCEPT ![t1] = @ \cup {t2}]
    /\ rwIn'  = [rwIn  EXCEPT ![t2] = @ \cup {t1}]
    /\ UNCHANGED txState

\* Commit a transaction only if it has no incoming-AND-outgoing rw edge
\* (the dangerous "pivot" pattern that closes a serialization cycle).
CommitOk(t) ==
    /\ txState[t] = "running"
    /\ ~ (rwIn[t] # {} /\ rwOut[t] # {})
    /\ txState' = [txState EXCEPT ![t] = "committed"]
    /\ UNCHANGED << rwOut, rwIn >>

\* Otherwise, abort the would-be pivot.
AbortPivot(t) ==
    /\ txState[t] = "running"
    /\ rwIn[t] # {} /\ rwOut[t] # {}
    /\ txState' = [txState EXCEPT ![t] = "aborted"]
    /\ UNCHANGED << rwOut, rwIn >>

Next == \/ \E t1, t2 \in Txns : RWConflict(t1, t2)
        \/ \E t \in Txns : CommitOk(t)
        \/ \E t \in Txns : AbortPivot(t)

Spec == Init /\ [][Next]_vars /\ WF_vars(Next)

\* Strong invariant: no committed pivot.
TypeOK == /\ txState \in [Txns -> States]
          /\ rwOut   \in [Txns -> SUBSET Txns]
          /\ rwIn    \in [Txns -> SUBSET Txns]
          /\ \A t \in Txns :
                txState[t] = "committed" => ~(rwIn[t] # {} /\ rwOut[t] # {})
====
