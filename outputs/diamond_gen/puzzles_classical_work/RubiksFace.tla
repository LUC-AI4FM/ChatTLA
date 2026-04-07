---- MODULE RubiksFace ----
(***************************************************************************)
(* A single face of a Rubik's cube modelled as a 3x3 array of colours.   *)
(* Three rotation actions: Rotate90, Rotate180, Rotate270 (clockwise).   *)
(* Strong invariant: the multiset of colours on the face is preserved by *)
(* every rotation.                                                        *)
(***************************************************************************)
EXTENDS Naturals, FiniteSets

VARIABLES face

vars == << face >>

N == 3
Cells == (1..N) \X (1..N)
Colors == 1..6

\* Rotation by 90 degrees clockwise: (r,c) <- (N+1-c, r)
Rot90(f)  == [c \in Cells |-> f[<< c[2], N + 1 - c[1] >>]]
Rot180(f) == [c \in Cells |-> f[<< N + 1 - c[1], N + 1 - c[2] >>]]
Rot270(f) == [c \in Cells |-> f[<< N + 1 - c[2], c[1] >>]]

\* Initial face: a fixed multiset of colours arranged so that all six occur.
Init ==
    face = [c \in Cells |->
              CASE c = << 1, 1 >> -> 1
                [] c = << 1, 2 >> -> 2
                [] c = << 1, 3 >> -> 3
                [] c = << 2, 1 >> -> 4
                [] c = << 2, 2 >> -> 5
                [] c = << 2, 3 >> -> 6
                [] c = << 3, 1 >> -> 1
                [] c = << 3, 2 >> -> 2
                [] OTHER          -> 3]

Rotate90  == face' = Rot90(face)
Rotate180 == face' = Rot180(face)
Rotate270 == face' = Rot270(face)

Next == Rotate90 \/ Rotate180 \/ Rotate270

Spec == Init /\ [][Next]_vars

CountColor(f, k) == Cardinality({ c \in Cells : f[c] = k })

\* The colour multiset of the initial face, computed once.
InitCount(k) ==
    CASE k = 1 -> 2
      [] k = 2 -> 2
      [] k = 3 -> 2
      [] k = 4 -> 1
      [] k = 5 -> 1
      [] OTHER -> 1

\* Strong invariant: the multiset of colours never changes.
SafetyInv == \A k \in Colors : CountColor(face, k) = InitCount(k)

TypeOK == /\ face \in [Cells -> Colors]
          /\ SafetyInv
====
