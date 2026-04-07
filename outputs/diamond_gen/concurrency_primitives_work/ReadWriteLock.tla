---- MODULE ReadWriteLock ----
EXTENDS Naturals, FiniteSets

CONSTANT Procs

\* writer is a SET of at most one process; empty means no writer.
VARIABLES readers, writer

vars == << readers, writer >>

Init == /\ readers = {}
        /\ writer  = {}

\* A reader may acquire only when no writer holds the lock.
AcquireRead(p) == /\ writer = {}
                  /\ p \notin readers
                  /\ readers' = readers \cup {p}
                  /\ UNCHANGED writer

ReleaseRead(p) == /\ p \in readers
                  /\ readers' = readers \ {p}
                  /\ UNCHANGED writer

\* A writer may acquire only when there are no readers and no other writer.
AcquireWrite(p) == /\ writer = {}
                   /\ readers = {}
                   /\ writer' = {p}
                   /\ UNCHANGED readers

ReleaseWrite(p) == /\ writer = {p}
                   /\ writer' = {}
                   /\ UNCHANGED readers

Next == \/ \E p \in Procs : AcquireRead(p)
        \/ \E p \in Procs : ReleaseRead(p)
        \/ \E p \in Procs : AcquireWrite(p)
        \/ \E p \in Procs : ReleaseWrite(p)

Spec == Init /\ [][Next]_vars

\* Safety: a writer excludes all readers (and any other writer).
RWSafe == /\ (writer = {}) \/ (readers = {})
          /\ Cardinality(writer) <= 1
          /\ (writer \cap readers) = {}

TypeOK == /\ readers \subseteq Procs
          /\ writer \subseteq Procs
          /\ RWSafe
====
