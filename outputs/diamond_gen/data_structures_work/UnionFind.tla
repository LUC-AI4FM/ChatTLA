---- MODULE UnionFind ----
EXTENDS Naturals, FiniteSets

CONSTANTS Universe

VARIABLES parent  \* parent[x] = canonical rep of x's component

vars == << parent >>

Init == parent = [x \in Universe |-> x]

\* Find: walk to a fixed point. With path compression flattened, parent[parent[x]] = parent[x].
Find(x) == parent[x]

\* Union: merge components by making one rep point to the other.
Union(a, b) == /\ Find(a) # Find(b)
               /\ LET ra == Find(a)
                      rb == Find(b)
                  IN parent' = [x \in Universe |->
                                  IF parent[x] = ra THEN rb ELSE parent[x]]

Next == \E a, b \in Universe : Union(a, b)

Spec == Init /\ [][Next]_vars /\ WF_vars(Next)

\* Strong invariant: parent points into the universe; reps are fixed points
\* (each rep's parent is itself), so components form a partition.
Reps == { x \in Universe : parent[x] = x }

Partition == /\ \A x \in Universe : parent[x] \in Universe
             /\ \A x \in Universe : parent[parent[x]] = parent[x]
             /\ \A x \in Universe : parent[x] \in Reps

TypeOK == /\ parent \in [Universe -> Universe]
          /\ Partition
====
