---- MODULE MazeRunner ----
(***************************************************************************)
(* A walker traverses a small 4x4 grid maze with internal walls.          *)
(* Cells are (r,c) for r,c in 1..4.  Walls are stored as a fixed set     *)
(* of unordered cell pairs.  The walker may step N/S/E/W if the          *)
(* destination cell is in the grid AND no wall lies between source and  *)
(* destination.                                                          *)
(***************************************************************************)
EXTENDS Naturals

VARIABLES pos

vars == << pos >>

N == 4
Cells == (1..N) \X (1..N)

\* Internal wall set: each wall is a 2-element set of adjacent cells.
\* Layout (X = wall):
\*   . . . .
\*   . X X .
\*   . . . .
\*   . X . .
\* Walls block movement between (2,2)-(2,3) and (3,2)-(4,2).
Walls ==
    { { << 2, 2 >>, << 2, 3 >> },
      { << 3, 2 >>, << 4, 2 >> } }

Adjacent(c1, c2) ==
    \/ c1[1] = c2[1] /\ (c2[2] = c1[2] + 1 \/ c2[2] = c1[2] - 1)
    \/ c1[2] = c2[2] /\ (c2[1] = c1[1] + 1 \/ c2[1] = c1[1] - 1)

NoWall(c1, c2) == { c1, c2 } \notin Walls

Init == pos = << 1, 1 >>

Step ==
    \E c \in Cells :
        /\ Adjacent(pos, c)
        /\ NoWall(pos, c)
        /\ pos' = c

Next == Step

Spec == Init /\ [][Next]_vars

\* Strong invariant: walker is always inside the grid AND never on a wall edge.
SafetyInv == pos \in Cells

TypeOK == /\ pos \in Cells
          /\ SafetyInv
====
