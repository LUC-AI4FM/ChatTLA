---- MODULE ReentrantMutex ----
EXTENDS Naturals, FiniteSets

CONSTANTS Procs, MaxDepth

\* owner is a set of at most one process; depth is the recursion depth held by it.
VARIABLES owner, depth

vars == << owner, depth >>

Init == /\ owner = {}
        /\ depth = 0

\* First acquire by anyone when free.
Acquire(p) == /\ owner = {}
              /\ owner' = {p}
              /\ depth' = 1

\* Reentrant acquire by current holder.
Reenter(p) == /\ owner = {p}
              /\ depth < MaxDepth
              /\ depth' = depth + 1
              /\ UNCHANGED owner

\* Release by current holder. depth-1 > 0 keeps holding; reaching 0 frees lock.
ReleasePartial(p) == /\ owner = {p}
                     /\ depth > 1
                     /\ depth' = depth - 1
                     /\ UNCHANGED owner

ReleaseFinal(p) == /\ owner = {p}
                   /\ depth = 1
                   /\ depth' = 0
                   /\ owner' = {}

Next == \/ \E p \in Procs : Acquire(p)
        \/ \E p \in Procs : Reenter(p)
        \/ \E p \in Procs : ReleasePartial(p)
        \/ \E p \in Procs : ReleaseFinal(p)

Spec == Init /\ [][Next]_vars

\* Safety: ownership and depth are consistent — depth>0 iff someone owns.
ReentrantSafe == /\ Cardinality(owner) <= 1
                 /\ ((owner = {}) <=> (depth = 0))
                 /\ depth >= 0

TypeOK == /\ owner \subseteq Procs
          /\ depth \in 0..MaxDepth
          /\ ReentrantSafe
====
