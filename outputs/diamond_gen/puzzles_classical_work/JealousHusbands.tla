---- MODULE JealousHusbands ----
(***************************************************************************)
(* Three jealous-husband couples must cross a river in a two-seat boat.    *)
(* Constraint: no wife may be on a bank with another husband unless her    *)
(* own husband is also on that bank.                                       *)
(*                                                                         *)
(* We track for each bank a set of "h1","h2","h3" (husbands) and           *)
(* "w1","w2","w3" (wives), plus boat side.  Couples are (h_i, w_i).        *)
(***************************************************************************)
EXTENDS Naturals, FiniteSets

VARIABLES left, right, boat

vars == << left, right, boat >>

Husbands == {"h1", "h2", "h3"}
Wives    == {"w1", "w2", "w3"}
People   == Husbands \cup Wives

\* Wife of husband h_i is w_i.
Mate(p) ==
    IF p = "h1" THEN "w1"
    ELSE IF p = "h2" THEN "w2"
    ELSE IF p = "h3" THEN "w3"
    ELSE IF p = "w1" THEN "h1"
    ELSE IF p = "w2" THEN "h2"
    ELSE "h3"

\* A bank is safe iff no wife is present alongside a husband other than her own.
SafeBank(b) ==
    \A w \in b \cap Wives :
        (Mate(w) \in b) \/ (b \cap Husbands = {})

Init == /\ left  = People
        /\ right = {}
        /\ boat  = "L"

\* Boat carries 1 or 2 people.
Loads(side) == { S \in SUBSET side : Cardinality(S) \in 1..2 }

CrossLR ==
    /\ boat = "L"
    /\ \E S \in Loads(left) :
          /\ left'  = left  \ S
          /\ right' = right \cup S
          /\ SafeBank(left')
          /\ SafeBank(right')
    /\ boat' = "R"

CrossRL ==
    /\ boat = "R"
    /\ \E S \in Loads(right) :
          /\ right' = right \ S
          /\ left'  = left  \cup S
          /\ SafeBank(left')
          /\ SafeBank(right')
    /\ boat' = "L"

Next == CrossLR \/ CrossRL

Spec == Init /\ [][Next]_vars

SafetyInv == SafeBank(left) /\ SafeBank(right)

TypeOK == /\ left  \subseteq People
          /\ right \subseteq People
          /\ left \cup right = People
          /\ left \cap right = {}
          /\ boat \in {"L", "R"}
          /\ SafetyInv
====
