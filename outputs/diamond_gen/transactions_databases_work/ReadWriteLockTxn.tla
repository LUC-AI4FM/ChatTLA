---- MODULE ReadWriteLockTxn ----
(***************************************************************************)
(*  Two-phase locking on a single shared resource.  Each transaction may  *)
(*  hold a Shared (S) or eXclusive (X) lock.  Compatibility matrix:       *)
(*                                                                         *)
(*                S         X                                              *)
(*           +----------------+                                            *)
(*         S | yes        no |                                             *)
(*         X | no         no |                                             *)
(*                                                                         *)
(*  Safety: at most one X holder, and X is incompatible with S.           *)
(***************************************************************************)
EXTENDS Naturals, FiniteSets

CONSTANTS Txns

VARIABLES sLocks, xLocks

vars == << sLocks, xLocks >>

Init == /\ sLocks = {}
        /\ xLocks = {}

\* A transaction may take an S lock as long as no X is held.
AcquireS(t) == /\ xLocks = {}
               /\ t \notin sLocks
               /\ t \notin xLocks
               /\ sLocks' = sLocks \cup {t}
               /\ UNCHANGED xLocks

\* A transaction may take an X lock only if it holds neither lock and no
\* other transaction holds any lock at all.
AcquireX(t) == /\ sLocks = {}
               /\ xLocks = {}
               /\ xLocks' = {t}
               /\ UNCHANGED sLocks

ReleaseS(t) == /\ t \in sLocks
               /\ sLocks' = sLocks \ {t}
               /\ UNCHANGED xLocks

ReleaseX(t) == /\ t \in xLocks
               /\ xLocks' = {}
               /\ UNCHANGED sLocks

Next == \/ \E t \in Txns : AcquireS(t)
        \/ \E t \in Txns : AcquireX(t)
        \/ \E t \in Txns : ReleaseS(t)
        \/ \E t \in Txns : ReleaseX(t)

Spec == Init /\ [][Next]_vars /\ WF_vars(Next)

\* Strong safety: writers exclude readers AND there is at most one writer.
\* Both clauses are conjoined into TypeOK.
TypeOK == /\ sLocks \subseteq Txns
          /\ xLocks \subseteq Txns
          /\ Cardinality(xLocks) <= 1
          /\ (xLocks = {} \/ sLocks = {})
          /\ sLocks \cap xLocks = {}
====
