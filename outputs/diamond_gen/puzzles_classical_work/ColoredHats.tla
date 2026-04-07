---- MODULE ColoredHats ----
(***************************************************************************)
(* The classic three-prisoner coloured-hat puzzle.                        *)
(* Three prisoners stand in a line.  Each wears a black or white hat.    *)
(* Prisoner i can see the hats of prisoners j > i, but not their own.    *)
(* Prisoners may guess "B", "W" or "?".  A guess is justified iff,      *)
(* given everything visible to that prisoner, there is no consistent    *)
(* assignment in which the opposite colour is correct.                  *)
(*                                                                         *)
(* In our small instance the hat assignment is fixed                    *)
(* (hats[1]=B, hats[2]=B, hats[3]=W) and exactly two of the three hats  *)
(* are black.  Each prisoner can therefore reason from the visible      *)
(* hats and from any earlier guess.                                     *)
(***************************************************************************)
EXTENDS Naturals

VARIABLES guess

vars == << guess >>

Prisoners == 1..3

\* Fixed but ground-truth-known assignment of hats: positions 1..3.
hats == [i \in Prisoners |->
            IF i = 3 THEN "W" ELSE "B"]

\* Total number of black hats in the world (a publicly known constant in
\* the puzzle setup).
TotalBlack == 2

\* Each prisoner sees only the hats of strictly greater indices.
Visible(i) == { j \in Prisoners : j > i }

\* Knowing the total number of black hats AND the colours of the visible
\* hats, prisoner i can deduce his own hat colour iff every prisoner j > i
\* has either announced a guess or is visible to him.  In our line model,
\* prisoner 3 sees nothing, prisoner 2 sees prisoner 3, prisoner 1 sees
\* prisoners 2 and 3.
SeenBlack(i) == IF Visible(i) = {} THEN 0
                ELSE IF Visible(i) = {3} THEN (IF hats[3] = "B" THEN 1 ELSE 0)
                ELSE (IF hats[2] = "B" THEN 1 ELSE 0)
                   + (IF hats[3] = "B" THEN 1 ELSE 0)

\* Prisoner 1 always knows: he sees both hats and knows the total.
Justified1 ==
    IF TotalBlack - SeenBlack(1) = 1 THEN "B" ELSE "W"

\* Prisoner 2 needs prisoner 1 to have already announced a guess so that
\* prisoner 2 can deduce his own hat from the missing information.
Justified2(g1) ==
    IF g1 = "?"
    THEN "?"
    ELSE
        \* Prisoner 1 deduced his colour; that pins down (g1 + hat3) so that
        \* the total black count is satisfied.  Prisoner 2 now knows hat2.
        LET unseen == TotalBlack
                       - (IF g1 = "B" THEN 1 ELSE 0)
                       - SeenBlack(2)
        IN  IF unseen = 1 THEN "B" ELSE "W"

Init == guess = [i \in Prisoners |-> "?"]

\* Prisoner 1 announces his (justified) guess.
Announce1 ==
    /\ guess[1] = "?"
    /\ guess' = [guess EXCEPT ![1] = Justified1]

\* Prisoner 2 announces only after prisoner 1 has spoken.
Announce2 ==
    /\ guess[2] = "?"
    /\ guess[1] /= "?"
    /\ guess' = [guess EXCEPT ![2] = Justified2(guess[1])]

\* Prisoner 3 sees nothing and never speaks in this small variant.
Idle ==
    /\ guess[1] /= "?" /\ guess[2] /= "?"
    /\ UNCHANGED guess

Next == Announce1 \/ Announce2 \/ Idle

Spec == Init /\ [][Next]_vars

\* Strong invariant: every announced guess matches the actual hat colour,
\* i.e. prisoners only speak when they can prove their answer.
SafetyInv ==
    \A i \in Prisoners :
        guess[i] = "?" \/ guess[i] = hats[i]

TypeOK == /\ guess \in [Prisoners -> {"?", "B", "W"}]
          /\ SafetyInv
====
