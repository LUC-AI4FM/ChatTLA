---- MODULE EightQueensSmall ----
(***************************************************************************)
(* Bounded N-queens.  Place at most K = 4 mutually non-attacking queens    *)
(* on a K x K board, one column at a time.  The state is the sequence of  *)
(* row positions chosen so far (one row per filled column).               *)
(*                                                                         *)
(* The strong safety invariant SafetyInv asserts that any two queens       *)
(* placed so far attack neither by row nor diagonal.  Columns are          *)
(* automatically distinct because at most one queen is placed per column. *)
(***************************************************************************)
EXTENDS Naturals, Sequences

VARIABLES queens   \* sequence of rows placed so far; column = index

vars == << queens >>

K == 4
Rows == 1..K

\* The queen at column i (placed at row queens[i]) does not attack one at
\* column j (row queens[j]) iff their rows differ AND their absolute row
\* difference is not equal to |i - j| (no diagonal attack).
AbsDiff(a, b) == IF a >= b THEN a - b ELSE b - a

NoAttack(s, i, j) ==
    /\ s[i] /= s[j]
    /\ AbsDiff(s[i], s[j]) /= AbsDiff(i, j)

NonAttacking(s) ==
    \A i \in 1..Len(s), j \in 1..Len(s) : (i < j) => NoAttack(s, i, j)

Init == queens = << >>

\* Place a queen on row r in the next free column, but only if it does not
\* attack any previously placed queen.
Place ==
    /\ Len(queens) < K
    /\ \E r \in Rows :
          /\ queens' = Append(queens, r)
          /\ NonAttacking(queens')

\* Idle when no further legal placement is possible (terminal "solution" state).
Done == /\ (Len(queens) = K \/ ~ \E r \in Rows : NonAttacking(Append(queens, r)))
        /\ UNCHANGED queens

Next == Place \/ Done

Spec == Init /\ [][Next]_vars

SafetyInv ==
    /\ Len(queens) <= K
    /\ NonAttacking(queens)

TypeOK == /\ queens \in Seq(Rows)
          /\ SafetyInv
====
