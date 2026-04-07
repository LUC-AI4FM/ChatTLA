---- MODULE WriteAheadLog ----
(***************************************************************************)
(*  Write-ahead log (WAL).  Each update is appended to the log first;    *)
(*  only after the log record is durable do we apply it to the data      *)
(*  page.  Recovery replays committed log records in order.              *)
(*                                                                         *)
(*  Strong invariant: the data state is always a prefix-application of   *)
(*  the log -- i.e. you cannot apply update i without having applied    *)
(*  every update j < i.                                                  *)
(***************************************************************************)
EXTENDS Naturals, Sequences

CONSTANTS MaxLog

VARIABLES log, applied

vars == << log, applied >>

Init == /\ log     = << >>
        /\ applied = 0

\* Append a record to the log (durable now).  We use natural number values.
LogWrite == /\ Len(log) < MaxLog
            /\ log' = Append(log, Len(log) + 1)
            /\ UNCHANGED applied

\* Apply the next undurable-but-applied prefix entry to the data state.
LogApply == /\ applied < Len(log)
            /\ applied' = applied + 1
            /\ UNCHANGED log

Next == \/ LogWrite \/ LogApply

Spec == Init /\ [][Next]_vars /\ WF_vars(Next)

\* Strong invariant: applied <= Len(log) and the log holds 1..Len(log)
\* (monotone, prefix property).  Conjoined into TypeOK so the mutation
\* test catches the lost prefix invariant.
TypeOK == /\ log     \in Seq(1..MaxLog)
          /\ applied \in 0..MaxLog
          /\ Len(log) <= MaxLog
          /\ applied <= Len(log)
          /\ \A i \in 1..Len(log) : log[i] = i
====
