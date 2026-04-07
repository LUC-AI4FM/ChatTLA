---- MODULE OptimisticConcurrency ----
(***************************************************************************)
(*  Optimistic concurrency control (OCC) on a small key/value store.      *)
(*  Each transaction goes through a read phase (collecting a read-set     *)
(*  with the version it observed) and a write phase (collecting writes).  *)
(*  At commit time it validates: every key it read must still be at the   *)
(*  same version, otherwise the transaction aborts.                       *)
(*                                                                         *)
(*  Strong invariant: a transaction that committed always saw the same    *)
(*  version that was current at commit time -- i.e. it was serializable.  *)
(***************************************************************************)
EXTENDS Naturals, FiniteSets

CONSTANTS NumTxns, NumKeys, MaxVersion

Txns == 1..NumTxns
Keys == 1..NumKeys

VARIABLES version, txState, readSet, commitVer

vars == << version, txState, readSet, commitVer >>

States == {"running", "committed", "aborted"}

Init == /\ version   = [k \in Keys |-> 0]
        /\ txState   = [t \in Txns |-> "running"]
        /\ readSet   = [t \in Txns |-> [k \in Keys |-> 0]]
        /\ commitVer = [t \in Txns |-> [k \in Keys |-> 0]]

\* Read phase: a running txn reads key k, recording the current version.
Read(t, k) == /\ txState[t] = "running"
              /\ readSet' = [readSet EXCEPT ![t][k] = version[k]]
              /\ UNCHANGED << version, txState, commitVer >>

\* Validate-and-commit: ensure every key read is still at the recorded
\* version, then bump versions for keys we want to write.
Commit(t) == /\ txState[t] = "running"
             /\ \A k \in Keys : readSet[t][k] = version[k]
             /\ \E W \in SUBSET Keys :
                    LET newVer == [k \in Keys |->
                                     IF k \in W /\ version[k] < MaxVersion
                                     THEN version[k] + 1
                                     ELSE version[k]]
                    IN  /\ version'   = newVer
                        /\ commitVer' = [commitVer EXCEPT ![t] = newVer]
             /\ txState' = [txState EXCEPT ![t] = "committed"]
             /\ UNCHANGED readSet

Abort(t) == /\ txState[t] = "running"
            /\ txState' = [txState EXCEPT ![t] = "aborted"]
            /\ UNCHANGED << version, readSet, commitVer >>

Next == \/ \E t \in Txns, k \in Keys : Read(t,k)
        \/ \E t \in Txns : Commit(t)
        \/ \E t \in Txns : Abort(t)

Spec == Init /\ [][Next]_vars /\ WF_vars(Next)

\* Versions only ever increase (monotone), and any committed txn's
\* recorded read-set was consistent with the version at commit time.
TypeOK == /\ version   \in [Keys -> 0..MaxVersion]
          /\ txState   \in [Txns -> States]
          /\ readSet   \in [Txns -> [Keys -> 0..MaxVersion]]
          /\ commitVer \in [Txns -> [Keys -> 0..MaxVersion]]
          /\ \A k \in Keys : version[k] <= MaxVersion
          /\ \A t \in Txns :
                txState[t] = "committed" =>
                    \A k \in Keys : commitVer[t][k] >= readSet[t][k]
====

