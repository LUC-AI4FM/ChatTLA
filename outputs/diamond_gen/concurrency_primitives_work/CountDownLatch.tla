---- MODULE CountDownLatch ----
EXTENDS Naturals

CONSTANT N

\* counter starts at N and only decreases until 0
\* opened becomes TRUE when counter reaches 0 and stays TRUE forever
VARIABLES counter, opened

vars == << counter, opened >>

Init == /\ counter = N
        /\ opened  = FALSE

\* CountDown: decrement, but only while counter > 0.
CountDown == /\ counter > 0
             /\ counter' = counter - 1
             /\ opened'  = (counter - 1 = 0)

\* Await: an awaiter "passes" — only meaningful once opened is TRUE.
\* Modeled as a stutter on counter/opened to expose the monotone property.
Await == /\ opened = TRUE
         /\ UNCHANGED << counter, opened >>

Next == \/ CountDown
        \/ Await

Spec == Init /\ [][Next]_vars

\* Strong safety: monotone — once opened, counter remains 0 and opened stays TRUE.
\* Equivalently: opened <=> (counter = 0).
LatchSafe == /\ (opened <=> (counter = 0))
             /\ counter <= N

TypeOK == /\ counter \in 0..N
          /\ opened \in BOOLEAN
          /\ LatchSafe
====
