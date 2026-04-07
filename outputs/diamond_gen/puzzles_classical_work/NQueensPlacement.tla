---- MODULE NQueensPlacement ----
(***************************************************************************)
(* Column-by-column N-queens placement.  We place exactly one queen in    *)
(* each successive column on a 4x4 board.  At every step the placement   *)
(* must remain non-attacking.  Variable col tracks the next column to    *)
(* fill.                                                                 *)
(***************************************************************************)
EXTENDS Naturals, Sequences

VARIABLES placement, col

vars == << placement, col >>

N == 4
Rows == 1..N

AbsDiff(a, b) == IF a >= b THEN a - b ELSE b - a

\* No two queens (in placement of length k) attack each other.
NonAttacking(p) ==
    \A i \in 1..Len(p), j \in 1..Len(p) :
        (i < j) => /\ p[i] /= p[j]
                   /\ AbsDiff(p[i], p[j]) /= AbsDiff(i, j)

Init == placement = << >> /\ col = 1

PlaceCol ==
    /\ col <= N
    /\ \E r \in Rows :
          /\ placement' = Append(placement, r)
          /\ NonAttacking(placement')
    /\ col' = col + 1

Done ==
    /\ \/ col > N
       \/ ~ \E r \in Rows : NonAttacking(Append(placement, r))
    /\ UNCHANGED << placement, col >>

Next == PlaceCol \/ Done

Spec == Init /\ [][Next]_vars

\* Strong invariant: column index matches queens placed AND no attacks.
SafetyInv ==
    /\ Len(placement) = col - 1
    /\ Len(placement) <= N
    /\ NonAttacking(placement)

TypeOK == /\ placement \in Seq(Rows)
          /\ col \in 1..(N + 1)
          /\ SafetyInv
====
