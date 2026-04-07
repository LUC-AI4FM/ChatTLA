---- MODULE RushHour ----
(***************************************************************************)
(* A miniature Rush Hour board: a 4x4 grid containing two cars.           *)
(* Car A is horizontal of length 2 (the "target" car).                    *)
(* Car B is vertical of length 3.                                         *)
(* Each car is represented by its head cell (smallest row,col).           *)
(* Cars slide along their axes by one cell, and never overlap.            *)
(***************************************************************************)
EXTENDS Integers

VARIABLES carA, carB   \* head cells: << row, col >>

vars == << carA, carB >>

N == 4
Cells == (1..N) \X (1..N)

\* Footprint of car A: head and head + (0,1).
CellsA(h) == { h, << h[1], h[2] + 1 >> }

\* Footprint of car B: head and head + (1,0) and head + (2,0).
CellsB(h) == { h, << h[1] + 1, h[2] >>, << h[1] + 2, h[2] >> }

InBoardA(h) == h[1] \in 1..N /\ h[2] \in 1..(N - 1)
InBoardB(h) == h[1] \in 1..(N - 2) /\ h[2] \in 1..N

NoOverlap(a, b) == CellsA(a) \cap CellsB(b) = {}

Init == /\ carA = << 1, 1 >>
        /\ carB = << 1, 4 >>

\* Slide A horizontally by +/- 1.
SlideA ==
    \E dc \in {-1, 1} :
        LET h2 == << carA[1], carA[2] + dc >>
        IN  /\ InBoardA(h2)
            /\ NoOverlap(h2, carB)
            /\ carA' = h2
            /\ carB' = carB

\* Slide B vertically by +/- 1.
SlideB ==
    \E dr \in {-1, 1} :
        LET h2 == << carB[1] + dr, carB[2] >>
        IN  /\ InBoardB(h2)
            /\ NoOverlap(carA, h2)
            /\ carB' = h2
            /\ carA' = carA

Next == SlideA \/ SlideB

Spec == Init /\ [][Next]_vars

\* Strong invariant: both cars stay on the board AND never overlap.
SafetyInv ==
    /\ InBoardA(carA)
    /\ InBoardB(carB)
    /\ NoOverlap(carA, carB)

TypeOK == /\ carA \in Cells
          /\ carB \in Cells
          /\ SafetyInv
====
