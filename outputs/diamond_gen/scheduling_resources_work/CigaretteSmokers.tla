---- MODULE CigaretteSmokers ----
(***************************************************************************)
(* The cigarette smokers problem (Patil, 1971).                            *)
(*                                                                         *)
(* Three smokers each have an infinite supply of one ingredient            *)
(* (tobacco / paper / matches).  An agent puts the OTHER TWO ingredients   *)
(* on the table; the smoker who has the missing third one then smokes.    *)
(*                                                                         *)
(* Safety: a smoker is "smoking" only when both ingredients it lacks are  *)
(* present on the table.                                                  *)
(***************************************************************************)
EXTENDS Naturals, FiniteSets

\* Smoker i owns ingredient i, lacks the other two.
Smokers == {0, 1, 2}
Ingredients == {0, 1, 2}

OwnedBy(s) == s              \* smoker s owns ingredient s
Lacks(s)   == Ingredients \ {s}

VARIABLES table, smoking

vars == << table, smoking >>

NoSmoker == 3

Init == /\ table = {}        \* nothing on the table
        /\ smoking = NoSmoker

\* Agent puts ingredients i and j on the table (the third smoker will smoke).
\* Only fires when the table is empty and no one is smoking.
AgentPut == /\ table = {}
            /\ smoking = NoSmoker
            /\ \E i, j \in Ingredients :
                 /\ i # j
                 /\ table' = {i, j}
                 /\ UNCHANGED smoking

\* Smoker s grabs both ingredients it lacks and starts smoking.
StartSmoke(s) == /\ smoking = NoSmoker
                 /\ Lacks(s) \subseteq table
                 /\ table' = table \ Lacks(s)
                 /\ smoking' = s

\* Finish smoking — table empty, smoker idle.
FinishSmoke == /\ smoking # NoSmoker
               /\ smoking' = NoSmoker
               /\ UNCHANGED table

Next == AgentPut \/ (\E s \in Smokers : StartSmoke(s)) \/ FinishSmoke

Spec == Init /\ [][Next]_vars

\* Strong safety: at most one smoker is smoking, and a smoker is only
\* smoking after having actually grabbed the missing ingredients
\* (encoded by the table being emptied of those ingredients).
SmokerInv == (smoking = NoSmoker) \/ (smoking \in Smokers /\ table = {})

TypeOK == /\ table \subseteq Ingredients /\ smoking \in Smokers \cup {NoSmoker} /\ SmokerInv
====
