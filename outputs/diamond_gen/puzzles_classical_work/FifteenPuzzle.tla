---- MODULE FifteenPuzzle ----
(***************************************************************************)
(* The 15-puzzle scaled down to 3x3 (the classic 8-puzzle).                *)
(* The board is a function from cells (i,j), i,j in 1..3, to tile values  *)
(* 0..8, with 0 representing the blank.  Each move slides a tile          *)
(* horizontally or vertically into the blank.                             *)
(*                                                                         *)
(* Strong invariant: tiles always form a permutation of 0..8 and the       *)
(* blank is the unique cell holding 0.                                     *)
(***************************************************************************)
EXTENDS Naturals, FiniteSets

VARIABLES grid, blank   \* grid : Cells -> 0..8 ; blank : Cells

vars == << grid, blank >>

Cells == (1..3) \X (1..3)
Tiles == 0..8

Adj(c1, c2) ==
    \/ c1[1] = c2[1] /\ (c1[2] = c2[2] + 1 \/ c1[2] = c2[2] - 1)
    \/ c1[2] = c2[2] /\ (c1[1] = c2[1] + 1 \/ c1[1] = c2[1] - 1)

Init ==
    /\ grid = [c \in Cells |->
                 ((c[1] - 1) * 3 + (c[2] - 1))]   \* tiles 0..8 in row-major order
    /\ blank = << 1, 1 >>

Slide ==
    \E c \in Cells :
        /\ Adj(blank, c)
        /\ grid' = [grid EXCEPT ![blank] = grid[c], ![c] = 0]
        /\ blank' = c

Next == Slide

Spec == Init /\ [][Next]_vars

\* Strong invariant: every tile value 0..8 appears exactly once and the
\* blank is the unique cell whose value is 0.
SafetyInv ==
    /\ \A t \in Tiles : Cardinality({ c \in Cells : grid[c] = t }) = 1
    /\ grid[blank] = 0

TypeOK == /\ grid \in [Cells -> Tiles]
          /\ blank \in Cells
          /\ SafetyInv
====
