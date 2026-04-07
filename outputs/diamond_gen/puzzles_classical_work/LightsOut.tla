---- MODULE LightsOut ----
(***************************************************************************)
(* The Lights Out puzzle on a 3x3 grid.  Each cell holds a 0/1 light.    *)
(* Pressing a cell toggles itself and its orthogonal neighbours.         *)
(* Corner presses toggle 3 cells, edges 4, the centre 5.                 *)
(***************************************************************************)
EXTENDS Naturals, FiniteSets

VARIABLES grid

vars == << grid >>

N == 3
Cells == (1..N) \X (1..N)

Toggled(c) ==
    { c } \cup
    ({ << c[1] - 1, c[2] >> } \cap Cells) \cup
    ({ << c[1] + 1, c[2] >> } \cap Cells) \cup
    ({ << c[1], c[2] - 1 >> } \cap Cells) \cup
    ({ << c[1], c[2] + 1 >> } \cap Cells)

Flip(b) == 1 - b

Init == grid = [c \in Cells |-> 0]

Press ==
    \E c \in Cells :
        grid' = [d \in Cells |->
                    IF d \in Toggled(c) THEN Flip(grid[d]) ELSE grid[d]]

Next == Press

Spec == Init /\ [][Next]_vars

\* Strong invariant: every cell holds 0 or 1, and the count of "on" lights
\* never exceeds the total number of cells.
OnCount == Cardinality({ c \in Cells : grid[c] = 1 })

SafetyInv ==
    /\ \A c \in Cells : grid[c] \in {0, 1}
    /\ OnCount <= N * N

TypeOK == /\ grid \in [Cells -> 0..1]
          /\ SafetyInv
====
