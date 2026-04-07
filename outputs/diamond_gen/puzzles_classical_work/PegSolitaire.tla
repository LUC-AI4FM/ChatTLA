---- MODULE PegSolitaire ----
(***************************************************************************)
(* A miniature peg solitaire board: a single row of seven holes.          *)
(* A move picks a peg p, an adjacent peg q (one step away in the same    *)
(* direction), and an empty cell e (the next step further), then jumps   *)
(* p over q into e, removing q.                                          *)
(*                                                                         *)
(* Initial position: pegs in cells 1..6, hole at cell 7.                 *)
(*                                                                         *)
(* Strong invariant: the peg count strictly equals (initial pegs minus   *)
(* number of moves) and stays within bounds.                             *)
(***************************************************************************)
EXTENDS Naturals, FiniteSets

VARIABLES board, moves   \* board: 1..N -> {0,1}; moves: count of jumps

vars == << board, moves >>

N == 7
Cells == 1..N

Init == /\ board = [c \in Cells |-> IF c < N THEN 1 ELSE 0]
        /\ moves = 0

\* Jump from p over q into e, where p,q,e are collinear, adjacent.
Jump ==
    \E p \in Cells, q \in Cells, e \in Cells :
        /\ board[p] = 1 /\ board[q] = 1 /\ board[e] = 0
        /\ \/ (q = p + 1 /\ e = p + 2)
           \/ (q = p - 1 /\ e = p - 2)
        /\ board' = [board EXCEPT ![p] = 0, ![q] = 0, ![e] = 1]
        /\ moves' = moves + 1

Done ==
    /\ ~ \E p \in Cells, q \in Cells, e \in Cells :
            /\ board[p] = 1 /\ board[q] = 1 /\ board[e] = 0
            /\ \/ (q = p + 1 /\ e = p + 2)
               \/ (q = p - 1 /\ e = p - 2)
    /\ UNCHANGED << board, moves >>

Next == Jump \/ Done

Spec == Init /\ [][Next]_vars

PegCount == Cardinality({ c \in Cells : board[c] = 1 })

\* Strong invariant: each move removes exactly one peg, so
\* PegCount + moves = (N - 1) (the original peg count).
SafetyInv ==
    /\ moves \in 0..(N - 1)
    /\ PegCount + moves = N - 1
    /\ PegCount >= 1

TypeOK == /\ board \in [Cells -> 0..1]
          /\ moves \in Nat
          /\ SafetyInv
====
