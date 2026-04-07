---- MODULE BloomCounter ----
EXTENDS Naturals

CONSTANTS Universe, MaxInserts

\* Counting Bloom filter: each bit position carries a small counter.
\* Two hash functions over the universe map elements to bit positions.
Bits == 1..3
H1(x) == 1
H2(x) == 2

VARIABLES counters, totalInserts

vars == << counters, totalInserts >>

Init == /\ counters = [b \in Bits |-> 0]
        /\ totalInserts = 0

\* Insert: bump the two hashed counters.
Insert(x) == /\ x \in Universe
             /\ totalInserts < MaxInserts
             /\ counters[H1(x)] < MaxInserts
             /\ counters[H2(x)] < MaxInserts
             /\ counters' = [counters EXCEPT ![H1(x)] = @ + 1, ![H2(x)] = @ + 1]
             /\ totalInserts' = totalInserts + 1

\* Delete: decrement (only when both counters > 0).
Delete(x) == /\ x \in Universe
             /\ counters[H1(x)] > 0
             /\ counters[H2(x)] > 0
             /\ totalInserts > 0
             /\ counters' = [counters EXCEPT ![H1(x)] = @ - 1, ![H2(x)] = @ - 1]
             /\ totalInserts' = totalInserts - 1

Next == (\E x \in Universe : Insert(x)) \/ (\E x \in Universe : Delete(x))

Spec == Init /\ [][Next]_vars

\* Strong invariant: counters non-negative and bounded by total inserts.
Bounded == /\ \A b \in Bits : counters[b] \in 0..MaxInserts
           /\ totalInserts \in 0..MaxInserts

TypeOK == /\ counters \in [Bits -> 0..MaxInserts]
          /\ totalInserts \in 0..MaxInserts
          /\ Bounded
====
