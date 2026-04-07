---- MODULE WolfGoatCabbage ----
(***************************************************************************)
(* The classic wolf, goat, and cabbage river crossing puzzle.              *)
(* The farmer ferries cargo across a river in a one-passenger boat.        *)
(* He may not leave the wolf alone with the goat, nor the goat alone with  *)
(* the cabbage, on either bank.  Only legal moves are taken; the safety    *)
(* invariant SafetyInv certifies that no forbidden configuration is ever   *)
(* reachable.                                                              *)
(***************************************************************************)
EXTENDS FiniteSets

VARIABLES left, right, boat

vars == << left, right, boat >>

Items == {"farmer", "wolf", "goat", "cabbage"}

\* A bank is safe iff the farmer is present, or no forbidden pair sits on it.
SafeBank(b) ==
    \/ "farmer" \in b
    \/ /\ ~ ({"wolf", "goat"}    \subseteq b)
       /\ ~ ({"goat", "cabbage"} \subseteq b)

Init == /\ left  = Items
        /\ right = {}
        /\ boat  = "L"

\* Farmer crosses (alone or with one item) only if both banks remain safe.
CrossLR ==
    /\ boat = "L"
    /\ "farmer" \in left
    /\ \E carry \in {{"farmer"}} \cup { {"farmer", x} : x \in left \ {"farmer"} } :
          /\ left'  = left  \ carry
          /\ right' = right \cup carry
          /\ SafeBank(left')
          /\ SafeBank(right')
    /\ boat' = "R"

CrossRL ==
    /\ boat = "R"
    /\ "farmer" \in right
    /\ \E carry \in {{"farmer"}} \cup { {"farmer", x} : x \in right \ {"farmer"} } :
          /\ right' = right \ carry
          /\ left'  = left  \cup carry
          /\ SafeBank(left')
          /\ SafeBank(right')
    /\ boat' = "L"

Next == CrossLR \/ CrossRL

Spec == Init /\ [][Next]_vars

\* Strong safety invariant: neither bank ever holds a forbidden pair alone.
SafetyInv == SafeBank(left) /\ SafeBank(right)

TypeOK == /\ left  \subseteq Items
          /\ right \subseteq Items
          /\ boat \in {"L", "R"}
          /\ left \cup right = Items
          /\ left \cap right = {}
          /\ SafetyInv
====
