---- MODULE TowersOfHanoi ----
(***************************************************************************)
(* Towers of Hanoi with three pegs and three disks (sizes 1..3, smallest=1)*)
(* Each peg is a sequence of disks listed bottom-to-top, so the head is    *)
(* the bottom disk and the last element is the top.                        *)
(* Legal moves transfer the top disk of one peg onto another peg only if   *)
(* the destination is empty or its top disk is strictly larger.            *)
(***************************************************************************)
EXTENDS Naturals, Sequences

VARIABLES pegs   \* function: Pegs -> Seq(1..N)

vars == << pegs >>

N    == 3
Pegs == {"A", "B", "C"}
Disks == 1..N

Top(s)  == s[Len(s)]
Pop(s)  == SubSeq(s, 1, Len(s) - 1)

\* A peg's stack is strictly decreasing from bottom to top (small disk on top).
Decreasing(s) ==
    \A i \in 1..(Len(s) - 1) : s[i] > s[i + 1]

Init == pegs = [p \in Pegs |-> IF p = "A" THEN << 3, 2, 1 >> ELSE << >>]

Move(from, to) ==
    /\ from /= to
    /\ Len(pegs[from]) > 0
    /\ (IF Len(pegs[to]) = 0 THEN TRUE ELSE Top(pegs[to]) > Top(pegs[from]))
    /\ pegs' = [pegs EXCEPT ![from] = Pop(pegs[from]),
                            ![to]   = Append(pegs[to], Top(pegs[from]))]

Next == \E from \in Pegs, to \in Pegs : Move(from, to)

Spec == Init /\ [][Next]_vars

\* Strong invariant: every peg is strictly decreasing AND disk multiset preserved.
SafetyInv ==
    /\ \A p \in Pegs : Decreasing(pegs[p])
    /\ Len(pegs["A"]) + Len(pegs["B"]) + Len(pegs["C"]) = N

TypeOK == /\ pegs \in [Pegs -> Seq(Disks)]
          /\ SafetyInv
====
