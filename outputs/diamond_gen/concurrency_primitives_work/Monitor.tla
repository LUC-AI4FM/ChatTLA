---- MODULE Monitor ----
EXTENDS Naturals, FiniteSets

CONSTANT Procs

\* inside  : the set of processes currently inside the monitor (at most one).
\* waiting : processes blocked on the condition variable.
VARIABLES inside, waiting

vars == << inside, waiting >>

Init == /\ inside  = {}
        /\ waiting = {}

\* Enter the monitor when nobody is inside and you are not waiting.
Enter(p) == /\ inside = {}
            /\ p \notin waiting
            /\ inside' = {p}
            /\ UNCHANGED waiting

\* Leave the monitor (no signaling).
Leave(p) == /\ inside = {p}
            /\ inside' = {}
            /\ UNCHANGED waiting

\* wait(): release the monitor and join the wait queue.
Wait(p) == /\ inside = {p}
           /\ inside' = {}
           /\ waiting' = waiting \cup {p}

\* signal(): wake one waiter — that waiter must re-enter the monitor.
\* Hoare semantics: the signaler must NOT be inside when wakeup transfers control.
\* We model this as: caller exits, woken waiter takes the monitor.
SignalAndExit(p) == /\ inside = {p}
                    /\ waiting # {}
                    /\ \E q \in waiting :
                          /\ inside' = {q}
                          /\ waiting' = waiting \ {q}

\* When the monitor is free, an external thread may pull a waiter back in.
\* This avoids deadlock when no thread holds the monitor to do the signaling.
WakeWaiter == /\ inside = {}
              /\ waiting # {}
              /\ \E q \in waiting :
                    /\ inside' = {q}
                    /\ waiting' = waiting \ {q}

Next == \/ \E p \in Procs : Enter(p)
        \/ \E p \in Procs : Leave(p)
        \/ \E p \in Procs : Wait(p)
        \/ \E p \in Procs : SignalAndExit(p)
        \/ WakeWaiter

Spec == Init /\ [][Next]_vars

\* Safety: at most one process inside the monitor; the inside process is not waiting.
MonitorSafe == /\ Cardinality(inside) <= 1
               /\ (inside \cap waiting) = {}

TypeOK == /\ inside  \subseteq Procs
          /\ waiting \subseteq Procs
          /\ MonitorSafe
====
