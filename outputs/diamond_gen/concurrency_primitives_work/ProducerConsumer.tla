---- MODULE ProducerConsumer ----
EXTENDS Naturals

CONSTANTS K, MaxItems

\* buffer    : current items in the bounded buffer (just a count)
\* produced  : total items produced so far
\* consumed  : total items consumed so far
VARIABLES buffer, produced, consumed

vars == << buffer, produced, consumed >>

Init == /\ buffer = 0
        /\ produced = 0
        /\ consumed = 0

\* Producer puts an item; blocks when buffer = K.
Produce == /\ buffer < K
           /\ produced < MaxItems
           /\ buffer'   = buffer + 1
           /\ produced' = produced + 1
           /\ UNCHANGED consumed

\* Consumer takes an item; blocks when buffer = 0.
Consume == /\ buffer > 0
           /\ buffer'   = buffer - 1
           /\ consumed' = consumed + 1
           /\ UNCHANGED produced

\* Reset to allow continued exploration once we hit the produce limit.
Reset == /\ produced = MaxItems
         /\ buffer = 0
         /\ consumed = MaxItems
         /\ produced' = 0
         /\ consumed' = 0
         /\ UNCHANGED buffer

Next == \/ Produce
        \/ Consume
        \/ Reset

Spec == Init /\ [][Next]_vars

\* Safety: 0 <= buffer <= K, and produced - consumed = buffer.
PCSafe == /\ buffer \in 0..K
          /\ produced - consumed = buffer
          /\ consumed <= produced

TypeOK == /\ buffer   \in 0..K
          /\ produced \in 0..MaxItems
          /\ consumed \in 0..MaxItems
          /\ PCSafe
====
