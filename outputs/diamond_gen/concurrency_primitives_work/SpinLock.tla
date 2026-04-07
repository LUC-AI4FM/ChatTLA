---- MODULE SpinLock ----
EXTENDS Naturals, FiniteSets

CONSTANT Procs

\* lock    : 0 = free, 1 = held
\* holder  : the unique holder when lock = 1; empty when free
\* spinning: set of processes currently spinning trying to CAS the lock
VARIABLES lock, holder, spinning

vars == << lock, holder, spinning >>

Init == /\ lock = 0
        /\ holder = {}
        /\ spinning = {}

\* Begin spinning to acquire the lock.
StartSpin(p) == /\ p \notin spinning
                /\ p \notin holder
                /\ spinning' = spinning \cup {p}
                /\ UNCHANGED << lock, holder >>

\* Successful test-and-set: only one spinning process wins.
TestAndSet(p) == /\ p \in spinning
                 /\ lock = 0
                 /\ lock' = 1
                 /\ holder' = {p}
                 /\ spinning' = spinning \ {p}

\* Release: holder writes 0.
Release(p) == /\ holder = {p}
              /\ lock = 1
              /\ lock' = 0
              /\ holder' = {}
              /\ UNCHANGED spinning

Next == \/ \E p \in Procs : StartSpin(p)
        \/ \E p \in Procs : TestAndSet(p)
        \/ \E p \in Procs : Release(p)

Spec == Init /\ [][Next]_vars

\* Strong safety: at most one holder; lock = 1 iff someone holds; holder
\* and spinning are disjoint.
SpinSafe == /\ Cardinality(holder) <= 1
            /\ ((lock = 1) <=> (holder # {}))
            /\ (holder \cap spinning) = {}

TypeOK == /\ lock \in 0..1
          /\ holder   \subseteq Procs
          /\ spinning \subseteq Procs
          /\ SpinSafe
====
