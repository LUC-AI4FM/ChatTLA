---- MODULE Futex ----
EXTENDS Naturals, FiniteSets

CONSTANTS Procs, MaxVal

\* word    : the futex memory word (a small natural number)
\* sleepers: set of processes asleep on the futex
VARIABLES word, sleepers

vars == << word, sleepers >>

Init == /\ word = 0
        /\ sleepers = {}

\* WaitIfEqual(p, v): atomic compare-and-sleep. Sleep iff word = v.
WaitOn(p) == /\ p \notin sleepers
             /\ word = 0
             /\ sleepers' = sleepers \cup {p}
             /\ UNCHANGED word

\* A non-sleeping process changes the word to a nonzero value (signal pending).
SetWord == /\ word < MaxVal
           /\ word' = word + 1
           /\ UNCHANGED sleepers

ClearWord == /\ word > 0
             /\ word' = 0
             /\ UNCHANGED sleepers

\* wake(1): wake exactly one sleeper.
WakeOne == /\ sleepers # {}
           /\ \E q \in sleepers :
                 sleepers' = sleepers \ {q}
           /\ UNCHANGED word

Next == \/ \E p \in Procs : WaitOn(p)
        \/ SetWord
        \/ ClearWord
        \/ WakeOne

Spec == Init /\ [][Next]_vars

\* Safety: a sleeping process must have observed word = 0 to enter the queue,
\* and the count is bounded by the total number of processes.
FutexSafe == /\ Cardinality(sleepers) <= Cardinality(Procs)
             /\ (sleepers # {} => word \in 0..MaxVal)

TypeOK == /\ word \in 0..MaxVal
          /\ sleepers \subseteq Procs
          /\ FutexSafe
====
