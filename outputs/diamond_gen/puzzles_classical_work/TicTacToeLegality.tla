---- MODULE TicTacToeLegality ----
(***************************************************************************)
(* Tic-tac-toe move legality.  Players X and O alternate placing marks    *)
(* on a 3x3 board.  X moves first.  We do not check for a winner — only   *)
(* legality of moves: marks land on empty cells, players strictly         *)
(* alternate, and the game stops as soon as the board is full.            *)
(***************************************************************************)
EXTENDS Naturals, FiniteSets

VARIABLES board, turn

vars == << board, turn >>

Cells  == (1..3) \X (1..3)
Marks  == {"X", "O", "_"}

CountMark(b, m) == Cardinality({ c \in Cells : b[c] = m })

Init == /\ board = [c \in Cells |-> "_"]
        /\ turn  = "X"

PlayMark(m) ==
    /\ turn = m
    /\ \E c \in Cells :
          /\ board[c] = "_"
          /\ board' = [board EXCEPT ![c] = m]
    /\ turn' = (IF m = "X" THEN "O" ELSE "X")

\* Idle when board is full so the spec does not deadlock.
Done ==
    /\ \A c \in Cells : board[c] /= "_"
    /\ UNCHANGED << board, turn >>

Next == PlayMark("X") \/ PlayMark("O") \/ Done

Spec == Init /\ [][Next]_vars

\* Strong invariant: turn is consistent with mark counts.
\* If it's X's turn then X and O have the same number of marks.
\* If it's O's turn then X has exactly one more mark than O.
SafetyInv ==
    /\ CountMark(board, "X") + CountMark(board, "O") + CountMark(board, "_") = 9
    /\ \/ /\ turn = "X"
          /\ CountMark(board, "X") = CountMark(board, "O")
       \/ /\ turn = "O"
          /\ CountMark(board, "X") = CountMark(board, "O") + 1

TypeOK == /\ board \in [Cells -> Marks]
          /\ turn \in {"X", "O"}
          /\ SafetyInv
====
