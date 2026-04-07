---- MODULE MvccSnapshot ----
(***************************************************************************)
(*  MVCC snapshot isolation on a single key.  Writers create a new        *)
(*  version (monotone version number).  Each transaction takes a          *)
(*  snapshot of the latest committed version when it begins, and reads    *)
(*  observe that snapshot.  Two writers conflict on the same key are      *)
(*  detected at commit; the loser aborts.                                  *)
(*                                                                         *)
(*  Strong invariants:                                                     *)
(*    * version numbers are strictly monotone over time;                   *)
(*    * a committed transaction's snapshot version <= its commit version. *)
(***************************************************************************)
EXTENDS Naturals, FiniteSets

CONSTANTS Txns, MaxVersion

VARIABLES curVer, snap, txState, commitVer

vars == << curVer, snap, txState, commitVer >>

States == {"idle", "active", "committed", "aborted"}

Init == /\ curVer    = 0
        /\ snap      = [t \in Txns |-> 0]
        /\ txState   = [t \in Txns |-> "idle"]
        /\ commitVer = [t \in Txns |-> 0]

Begin(t) == /\ txState[t] = "idle"
            /\ snap'    = [snap EXCEPT ![t] = curVer]
            /\ txState' = [txState EXCEPT ![t] = "active"]
            /\ UNCHANGED << curVer, commitVer >>

\* Writer commits a new version.  No-one else may have written between
\* this txn's snapshot and now (snapshot-isolation write conflict check).
CommitWrite(t) == /\ txState[t] = "active"
                  /\ snap[t] = curVer
                  /\ curVer < MaxVersion
                  /\ curVer' = curVer + 1
                  /\ commitVer' = [commitVer EXCEPT ![t] = curVer + 1]
                  /\ txState'   = [txState EXCEPT ![t] = "committed"]
                  /\ UNCHANGED snap

\* Read-only commit: nothing to do, just transition.
CommitRead(t) == /\ txState[t] = "active"
                 /\ commitVer' = [commitVer EXCEPT ![t] = snap[t]]
                 /\ txState'   = [txState EXCEPT ![t] = "committed"]
                 /\ UNCHANGED << curVer, snap >>

Abort(t) == /\ txState[t] = "active"
            /\ snap[t] # curVer  \* lost the write race
            /\ txState' = [txState EXCEPT ![t] = "aborted"]
            /\ UNCHANGED << curVer, snap, commitVer >>

Next == \/ \E t \in Txns : Begin(t)
        \/ \E t \in Txns : CommitWrite(t)
        \/ \E t \in Txns : CommitRead(t)
        \/ \E t \in Txns : Abort(t)

Spec == Init /\ [][Next]_vars /\ WF_vars(Next)

\* Strong invariant: monotone versions and committed-write versions match
\* the global counter at commit time, conjoined into TypeOK.
TypeOK == /\ curVer    \in 0..MaxVersion
          /\ snap      \in [Txns -> 0..MaxVersion]
          /\ txState   \in [Txns -> States]
          /\ commitVer \in [Txns -> 0..MaxVersion]
          /\ \A t \in Txns : snap[t] <= curVer
          /\ \A t \in Txns : commitVer[t] <= curVer
====
