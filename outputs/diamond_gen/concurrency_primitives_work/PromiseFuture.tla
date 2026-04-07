---- MODULE PromiseFuture ----
EXTENDS Naturals, FiniteSets

CONSTANTS Procs, MaxVal

\* state in {0=empty, 1=fulfilled}
\* value : the value once fulfilled, 0 means none
\* readers : the set of consumers that have read the value
VARIABLES state, value, readers

vars == << state, value, readers >>

Empty     == 0
Fulfilled == 1

Init == /\ state = Empty
        /\ value = 0
        /\ readers = {}

\* The producer fulfills the future exactly once with some value v in 1..MaxVal.
Fulfill == /\ state = Empty
           /\ \E v \in 1..MaxVal :
                 /\ value' = v
                 /\ state' = Fulfilled
           /\ UNCHANGED readers

\* A consumer reads the value (only after it has been fulfilled).
Read(p) == /\ state = Fulfilled
           /\ p \notin readers
           /\ readers' = readers \cup {p}
           /\ UNCHANGED << state, value >>

\* All consumers can leave the read region to avoid terminal deadlock.
ResetReaders == /\ state = Fulfilled
                /\ readers = Procs
                /\ readers' = {}
                /\ UNCHANGED << state, value >>

Next == \/ Fulfill
        \/ \E p \in Procs : Read(p)
        \/ ResetReaders

Spec == Init /\ [][Next]_vars

\* Safety: at most one fulfill (state monotone), and value is consistent
\* with state (value > 0 iff fulfilled).
PromiseSafe == /\ ((state = Empty)     => (value = 0))
               /\ ((state = Fulfilled) => (value \in 1..MaxVal))

TypeOK == /\ state \in {Empty, Fulfilled}
          /\ value \in 0..MaxVal
          /\ readers \subseteq Procs
          /\ PromiseSafe
====
