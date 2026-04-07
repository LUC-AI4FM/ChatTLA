---- MODULE WaitGroup ----
EXTENDS Naturals

CONSTANT MaxCount

\* counter : the number of outstanding tasks (Add increments, Done decrements)
\* released: TRUE once Wait has been observed to return (counter reached 0)
VARIABLES counter, released

vars == << counter, released >>

Init == /\ counter = 0
        /\ released = FALSE

\* Add(1): only legal while not released; bounded to MaxCount.
Add == /\ ~released
       /\ counter < MaxCount
       /\ counter' = counter + 1
       /\ UNCHANGED released

\* Done: decrement; legal only when counter > 0.
Done == /\ counter > 0
        /\ counter' = counter - 1
        /\ UNCHANGED released

\* Wait observes counter = 0 and unblocks.
WaitReturn == /\ counter = 0
              /\ ~released
              /\ released' = TRUE
              /\ UNCHANGED counter

\* Reset for next cycle (after Wait has returned and counter is 0).
Reset == /\ released
         /\ counter = 0
         /\ released' = FALSE
         /\ UNCHANGED counter

Next == \/ Add
        \/ Done
        \/ WaitReturn
        \/ Reset

Spec == Init /\ [][Next]_vars

\* Safety: counter never negative; once released, counter stays 0 until Reset.
WaitGroupSafe == /\ counter >= 0
                 /\ counter <= MaxCount
                 /\ (released => counter = 0)

TypeOK == /\ counter \in 0..MaxCount
          /\ released \in BOOLEAN
          /\ WaitGroupSafe
====
