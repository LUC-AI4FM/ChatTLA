---- MODULE ChessKingMoves ----
(***************************************************************************)
(* A single chess king moves on a 4x4 board.  Two enemy rooks sit on      *)
(* fixed squares and "attack" their entire row and column (excluding     *)
(* the king itself).  The king may step to any of the eight orthogonal   *)
(* or diagonal neighbours, but never onto an attacked square.            *)
(***************************************************************************)
EXTENDS Naturals

VARIABLES king

vars == << king >>

N == 4
Squares == (1..N) \X (1..N)

Rook1 == << 1, 4 >>
Rook2 == << 4, 1 >>

AbsDiff(a, b) == IF a >= b THEN a - b ELSE b - a

Adjacent(c1, c2) ==
    LET d1 == AbsDiff(c1[1], c2[1])
        d2 == AbsDiff(c1[2], c2[2])
    IN  d1 <= 1 /\ d2 <= 1 /\ (d1 + d2) > 0

\* A square is attacked by a rook on the same row or column.
AttackedBy(rook, sq) ==
    sq /= rook /\ (sq[1] = rook[1] \/ sq[2] = rook[2])

Safe(sq) ==
    /\ sq /= Rook1
    /\ sq /= Rook2
    /\ ~ AttackedBy(Rook1, sq)
    /\ ~ AttackedBy(Rook2, sq)

Init == king = << 2, 2 >>

Move ==
    \E sq \in Squares :
        /\ Adjacent(king, sq)
        /\ Safe(sq)
        /\ king' = sq

Done ==
    /\ ~ \E sq \in Squares : Adjacent(king, sq) /\ Safe(sq)
    /\ UNCHANGED king

Next == Move \/ Done

Spec == Init /\ [][Next]_vars

\* Strong invariant: the king always stands on a safe square inside the board.
SafetyInv == king \in Squares /\ Safe(king)

TypeOK == /\ king \in Squares
          /\ SafetyInv
====
