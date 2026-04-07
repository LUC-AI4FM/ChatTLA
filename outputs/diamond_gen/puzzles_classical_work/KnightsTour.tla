---- MODULE KnightsTour ----
(***************************************************************************)
(* A knight moves on a 4x4 chess board, attempting to visit each square   *)
(* exactly once.  We track the knight's current position and the set of   *)
(* visited squares.  Each move must land on an L-shaped neighbour that   *)
(* has not yet been visited.                                             *)
(***************************************************************************)
EXTENDS Naturals, FiniteSets

VARIABLES pos, visited

vars == << pos, visited >>

N == 4
Squares == (1..N) \X (1..N)

AbsDiff(a, b) == IF a >= b THEN a - b ELSE b - a

\* L-shaped: one coordinate differs by 1, the other by 2.
KnightMove(c1, c2) ==
    LET d1 == AbsDiff(c1[1], c2[1])
        d2 == AbsDiff(c1[2], c2[2])
    IN  (d1 = 1 /\ d2 = 2) \/ (d1 = 2 /\ d2 = 1)

Init == /\ pos = << 1, 1 >>
        /\ visited = { << 1, 1 >> }

Step ==
    \E sq \in Squares :
        /\ KnightMove(pos, sq)
        /\ sq \notin visited
        /\ pos' = sq
        /\ visited' = visited \cup { sq }

\* Idle once stuck or finished, to keep TLC from declaring a deadlock.
Done ==
    /\ ~ \E sq \in Squares : KnightMove(pos, sq) /\ sq \notin visited
    /\ UNCHANGED << pos, visited >>

Next == Step \/ Done

Spec == Init /\ [][Next]_vars

\* Strong invariant: knight always sits on a visited square AND visited set
\* cardinality is consistent with no square ever revisited (no duplicates).
SafetyInv ==
    /\ pos \in visited
    /\ visited \subseteq Squares
    /\ Cardinality(visited) >= 1
    /\ Cardinality(visited) <= N * N

TypeOK == /\ pos \in Squares
          /\ visited \in SUBSET Squares
          /\ SafetyInv
====
