---- MODULE FsmTrafficLight ----
(***************************************************************************)
(*  Traffic-light FSM cycling red -> green -> yellow -> red.               *)
(*  Track previous color so the safety invariant "yellow follows green"   *)
(*  is a checkable state predicate.                                       *)
(***************************************************************************)
EXTENDS Naturals

CONSTANT MaxTicks

VARIABLES color, prev_color, ticks

vars == << color, prev_color, ticks >>

Colors == {"red", "green", "yellow"}

Init == /\ color      = "red"
        /\ prev_color = "red"
        /\ ticks      = 0

ToGreen == /\ color = "red"
           /\ ticks < MaxTicks
           /\ color' = "green"
           /\ prev_color' = color
           /\ ticks' = ticks + 1

ToYellow == /\ color = "green"
            /\ ticks < MaxTicks
            /\ color' = "yellow"
            /\ prev_color' = color
            /\ ticks' = ticks + 1

ToRed == /\ color = "yellow"
         /\ ticks < MaxTicks
         /\ color' = "red"
         /\ prev_color' = color
         /\ ticks' = ticks + 1

Done == /\ ticks = MaxTicks
        /\ UNCHANGED vars

Next == \/ ToGreen \/ ToYellow \/ ToRed \/ Done

Spec == Init /\ [][Next]_vars

\* Yellow may only follow green; the cycle is strict and never skips a phase.
SafetyInvariant == ((color = "yellow") => (prev_color = "green")) /\ ((color = "green") => (prev_color = "red")) /\ (~(color = "green" /\ prev_color = "green"))

TypeOK == /\ color \in Colors
          /\ prev_color \in Colors
          /\ ticks \in 0..MaxTicks
          /\ SafetyInvariant
====
