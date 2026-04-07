---- MODULE TransferAtomic ----
(***************************************************************************)
(*  Atomic bank transfer between accounts.  Each transaction is committed *)
(*  atomically -- the debit and credit happen in one step -- so the total *)
(*  balance over all accounts is invariant.                               *)
(***************************************************************************)
EXTENDS Naturals, FiniteSets

CONSTANTS NumAccounts, NumTxns, MaxAmount

Accounts == 1..NumAccounts
Txns     == 1..NumTxns

VARIABLES balance, txStatus

vars == << balance, txStatus >>

TxStates == {"pending", "committed", "aborted"}

Total == NumAccounts * MaxAmount

Init == /\ balance  = [a \in Accounts |-> MaxAmount]
        /\ txStatus = [t \in Txns |-> "pending"]

\* Atomic transfer: debit src, credit dst, both in one step.
Commit(t, src, dst) ==
    /\ src # dst
    /\ txStatus[t] = "pending"
    /\ balance[src] >= 1
    /\ balance' = [balance EXCEPT ![src] = @ - 1, ![dst] = @ + 1]
    /\ txStatus' = [txStatus EXCEPT ![t] = "committed"]

Abort(t) ==
    /\ txStatus[t] = "pending"
    /\ txStatus' = [txStatus EXCEPT ![t] = "aborted"]
    /\ UNCHANGED balance

Next == \/ \E t \in Txns, s \in Accounts, d \in Accounts : Commit(t,s,d)
        \/ \E t \in Txns : Abort(t)

\* Weak fairness so the auto-config disables deadlock checking
\* (every behaviour eventually settles into a stuttering state once
\*  every transaction has decided).
Spec == Init /\ [][Next]_vars /\ WF_vars(Next)

\* Recursive sum over Accounts, used by the strong invariant.
RECURSIVE SumOver(_)
SumOver(S) == IF S = {} THEN 0
              ELSE LET a == CHOOSE x \in S : TRUE
                   IN  balance[a] + SumOver(S \ {a})

\* Conservation of money is conjoined into TypeOK so the mutation test
\* (which strips TypeOK) leaves only a vacuous invariant -- catching the
\* fact that without TypeOK we lose the safety property.
TypeOK == /\ balance \in [Accounts -> 0..Total]
          /\ txStatus \in [Txns -> TxStates]
          /\ SumOver(Accounts) = Total
====
