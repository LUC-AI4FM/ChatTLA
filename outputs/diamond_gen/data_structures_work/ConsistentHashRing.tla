---- MODULE ConsistentHashRing ----
EXTENDS Naturals, FiniteSets

CONSTANTS RingSize, NumNodes, NumKeys

\* Positions on the ring: 0..RingSize-1. Each node and key hashes to a position.
Positions == 0..(RingSize - 1)
NodeIds == 1..NumNodes
KeyIds == 1..NumKeys

\* Deterministic hashes (modeled as identity mod ring size).
NodeHash(n) == n % RingSize
KeyHash(k) == (k * 2) % RingSize

VARIABLES alive  \* set of currently alive nodes

vars == << alive >>

Init == alive = NodeIds

\* Successor of position p among alive nodes (the first alive node with hash >= p,
\* wrapping around). Returns NodeId.
Successor(p) ==
  LET cands == { n \in alive : NodeHash(n) >= p }
  IN IF cands # {} THEN CHOOSE n \in cands : \A m \in cands : NodeHash(n) <= NodeHash(m)
     ELSE CHOOSE n \in alive : \A m \in alive : NodeHash(n) <= NodeHash(m)

\* Owner of a key.
Owner(k) == Successor(KeyHash(k))

\* Mark a node as failed (must keep at least one alive).
Fail(n) == /\ n \in alive
           /\ Cardinality(alive) > 1
           /\ alive' = alive \ {n}

\* Recover a previously failed node.
Recover(n) == /\ n \in NodeIds
              /\ n \notin alive
              /\ alive' = alive \cup {n}

Next == (\E n \in NodeIds : Fail(n)) \/ (\E n \in NodeIds : Recover(n))

Spec == Init /\ [][Next]_vars

\* Strong invariant: every key maps to exactly one alive node.
EveryKeyMapped == /\ alive # {}
                  /\ \A k \in KeyIds : Owner(k) \in alive

TypeOK == /\ alive \subseteq NodeIds
          /\ EveryKeyMapped
====
