---- MODULE WaterJugs ----
(***************************************************************************)
(* The classic Die-Hard water-jug puzzle: a 5-liter and a 3-liter jug,     *)
(* with operations Fill, Empty and Pour, the goal is to measure 4 liters. *)
(* TLC checks that levels never overflow either jug.                       *)
(***************************************************************************)
EXTENDS Naturals

VARIABLES big, small

vars == << big, small >>

BigCap   == 5
SmallCap == 3

Min(a, b) == IF a <= b THEN a ELSE b

Init == big = 0 /\ small = 0

FillBig    == big' = BigCap /\ small' = small
FillSmall  == small' = SmallCap /\ big' = big
EmptyBig   == big' = 0 /\ small' = small
EmptySmall == small' = 0 /\ big' = big

\* Pour from small into big until either small is empty or big is full.
PourSmallToBig ==
    LET amount == Min(small, BigCap - big)
    IN  /\ amount > 0
        /\ big'   = big + amount
        /\ small' = small - amount

\* Pour from big into small symmetrically.
PourBigToSmall ==
    LET amount == Min(big, SmallCap - small)
    IN  /\ amount > 0
        /\ small' = small + amount
        /\ big'   = big - amount

Next ==
    \/ FillBig \/ FillSmall
    \/ EmptyBig \/ EmptySmall
    \/ PourSmallToBig \/ PourBigToSmall

Spec == Init /\ [][Next]_vars

\* Strong invariant: both levels stay within their jug capacities.
SafetyInv == big \in 0..BigCap /\ small \in 0..SmallCap

TypeOK == /\ big \in Nat
          /\ small \in Nat
          /\ SafetyInv
====
