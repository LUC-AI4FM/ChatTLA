---- MODULE Sudoku4x4 ----
(***************************************************************************)
(* A mini 4x4 Sudoku.  Cells take values 1..4 with 0 meaning blank.       *)
(* Each move fills one blank cell with a digit, but only if doing so      *)
(* keeps the partial assignment valid: no row, column, or 2x2 region      *)
(* contains a duplicate digit.                                            *)
(***************************************************************************)
EXTENDS Naturals, FiniteSets

VARIABLES grid

vars == << grid >>

N      == 4
Rows   == 1..N
Cols   == 1..N
Cells  == Rows \X Cols
Digits == 1..N

\* The 2x2 box containing cell (r,c).  Boxes indexed (1..2) x (1..2).
Box(r, c) ==
    LET br == ((r - 1) \div 2) + 1
        bc == ((c - 1) \div 2) + 1
    IN  { << i, j >> \in Cells :
              ((i - 1) \div 2) + 1 = br /\ ((j - 1) \div 2) + 1 = bc }

NoDup(s) ==
    \A c1 \in s, c2 \in s :
        (c1 /= c2 /\ grid[c1] /= 0 /\ grid[c2] /= 0) => grid[c1] /= grid[c2]

NoDupAfter(s, cell, d) ==
    \A c \in s :
        (c /= cell /\ grid[c] /= 0) => grid[c] /= d

ValidGrid ==
    /\ \A r \in Rows : NoDup({ << r, c >> : c \in Cols })
    /\ \A c \in Cols : NoDup({ << r, c >> : r \in Rows })
    /\ \A r \in {1, 3}, c \in {1, 3} : NoDup(Box(r, c))

\* Initial puzzle: a nearly-complete legal grid with three blanks to fill.
\* Solution layout:
\*   1 2 | 3 4
\*   3 4 | 1 2
\*   ----+----
\*   2 1 | 4 3
\*   4 3 | 2 1
Init ==
    grid = [cell \in Cells |->
              CASE cell = << 1, 1 >> -> 1
                [] cell = << 1, 2 >> -> 2
                [] cell = << 1, 3 >> -> 3
                [] cell = << 1, 4 >> -> 4
                [] cell = << 2, 1 >> -> 3
                [] cell = << 2, 2 >> -> 4
                [] cell = << 2, 3 >> -> 1
                [] cell = << 2, 4 >> -> 2
                [] cell = << 3, 1 >> -> 2
                [] cell = << 3, 2 >> -> 1
                [] cell = << 3, 3 >> -> 0   \* blank
                [] cell = << 3, 4 >> -> 3
                [] cell = << 4, 1 >> -> 4
                [] cell = << 4, 2 >> -> 0   \* blank
                [] cell = << 4, 3 >> -> 2
                [] cell = << 4, 4 >> -> 0]  \* blank

Fill ==
    \E cell \in Cells, d \in Digits :
        /\ grid[cell] = 0
        /\ NoDupAfter({ << cell[1], j >> : j \in Cols }, cell, d)
        /\ NoDupAfter({ << i, cell[2] >> : i \in Rows }, cell, d)
        /\ NoDupAfter(Box(cell[1], cell[2]), cell, d)
        /\ grid' = [grid EXCEPT ![cell] = d]

Done ==
    /\ \A cell \in Cells : grid[cell] /= 0
    /\ UNCHANGED grid

Next == Fill \/ Done

Spec == Init /\ [][Next]_vars

\* Strong invariant: the partial grid is always Sudoku-legal.
SafetyInv == ValidGrid

TypeOK == /\ grid \in [Cells -> 0..N]
          /\ SafetyInv
====
