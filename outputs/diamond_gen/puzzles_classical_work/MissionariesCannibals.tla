---- MODULE MissionariesCannibals ----
(***************************************************************************)
(* Three missionaries and three cannibals must cross a river in a boat     *)
(* that holds at most two.  On either bank, cannibals must never           *)
(* outnumber missionaries (when missionaries are present), or the          *)
(* missionaries get eaten.                                                 *)
(***************************************************************************)
EXTENDS Naturals

VARIABLES mL, cL, mR, cR, boat

vars == << mL, cL, mR, cR, boat >>

N == 3   \* three of each

\* A side is safe if there are no missionaries or cannibals do not outnumber them.
SideSafe(m, c) == (m = 0) \/ (c <= m)

Init == /\ mL = N /\ cL = N
        /\ mR = 0 /\ cR = 0
        /\ boat = "L"

\* Boat carries (dm, dc) people: 1 <= dm + dc <= 2.
MoveLR ==
    /\ boat = "L"
    /\ \E dm \in 0..2, dc \in 0..2 :
          /\ dm + dc >= 1
          /\ dm + dc <= 2
          /\ dm <= mL /\ dc <= cL
          /\ mL' = mL - dm /\ cL' = cL - dc
          /\ mR' = mR + dm /\ cR' = cR + dc
          /\ SideSafe(mL', cL')
          /\ SideSafe(mR', cR')
    /\ boat' = "R"

MoveRL ==
    /\ boat = "R"
    /\ \E dm \in 0..2, dc \in 0..2 :
          /\ dm + dc >= 1
          /\ dm + dc <= 2
          /\ dm <= mR /\ dc <= cR
          /\ mR' = mR - dm /\ cR' = cR - dc
          /\ mL' = mL + dm /\ cL' = cL + dc
          /\ SideSafe(mL', cL')
          /\ SideSafe(mR', cR')
    /\ boat' = "L"

Next == MoveLR \/ MoveRL

Spec == Init /\ [][Next]_vars

\* Strong invariant: counts conserved AND no missionary ever outnumbered.
SafetyInv ==
    /\ mL + mR = N
    /\ cL + cR = N
    /\ SideSafe(mL, cL)
    /\ SideSafe(mR, cR)

TypeOK == /\ mL \in 0..N /\ cL \in 0..N
          /\ mR \in 0..N /\ cR \in 0..N
          /\ boat \in {"L", "R"}
          /\ SafetyInv
====
