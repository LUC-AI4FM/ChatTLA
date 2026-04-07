---- MODULE RiverCrossingFamily ----
(***************************************************************************)
(* Family river-crossing puzzle.  Five people must cross:                 *)
(*   "father", "mother", "child1", "child2", "police", "thief".          *)
(* The boat seats two and needs an adult driver ("father", "mother", or  *)
(* "police").  The thief may not be alone with any family member         *)
(* unless the policeman is on the same bank.                             *)
(***************************************************************************)
EXTENDS Naturals, FiniteSets

VARIABLES left, right, boat

vars == << left, right, boat >>

People == {"father", "mother", "child1", "child2", "police", "thief"}
Adults == {"father", "mother", "police"}
Family == {"father", "mother", "child1", "child2"}

\* A bank is safe iff:
\*   - The thief is absent, OR
\*   - The policeman is on it, OR
\*   - No family member is on it.
SafeBank(b) ==
    \/ "thief"  \notin b
    \/ "police" \in b
    \/ b \cap Family = {}

Init == /\ left  = People
        /\ right = {}
        /\ boat  = "L"

\* The boat seats 1 or 2 people, and at least one must be an adult.
ValidLoad(S) ==
    /\ Cardinality(S) \in 1..2
    /\ S \cap Adults /= {}

CrossLR ==
    /\ boat = "L"
    /\ \E S \in SUBSET left :
          /\ ValidLoad(S)
          /\ left'  = left \ S
          /\ right' = right \cup S
          /\ SafeBank(left')
          /\ SafeBank(right')
    /\ boat' = "R"

CrossRL ==
    /\ boat = "R"
    /\ \E S \in SUBSET right :
          /\ ValidLoad(S)
          /\ right' = right \ S
          /\ left'  = left \cup S
          /\ SafeBank(left')
          /\ SafeBank(right')
    /\ boat' = "L"

Next == CrossLR \/ CrossRL

Spec == Init /\ [][Next]_vars

SafetyInv ==
    /\ left \cup right = People
    /\ left \cap right = {}
    /\ SafeBank(left)
    /\ SafeBank(right)

TypeOK == /\ left \subseteq People
          /\ right \subseteq People
          /\ boat \in {"L", "R"}
          /\ SafetyInv
====
