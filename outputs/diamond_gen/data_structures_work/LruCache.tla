---- MODULE LruCache ----
EXTENDS Naturals, Sequences, FiniteSets

CONSTANTS K, Keys

VARIABLES order  \* sequence of distinct keys; head = LRU, tail = MRU

vars == << order >>

\* True iff key k appears in seq s.
InSeq(s, k) == \E i \in 1..Len(s) : s[i] = k

\* Remove first occurrence of k from sequence s.
RemoveKey(s, k) ==
  LET idx == CHOOSE i \in 1..Len(s) : s[i] = k
  IN SubSeq(s, 1, idx - 1) \o SubSeq(s, idx + 1, Len(s))

Init == order = << >>

\* Get on a hit: bump key to MRU position.
Get(k) == /\ InSeq(order, k)
          /\ order' = Append(RemoveKey(order, k), k)

\* Put: insert as MRU; if at capacity, evict LRU (head); if already present, refresh.
Put(k) == /\ k \in Keys
          /\ \/ /\ InSeq(order, k)
                /\ order' = Append(RemoveKey(order, k), k)
             \/ /\ ~InSeq(order, k)
                /\ Len(order) < K
                /\ order' = Append(order, k)
             \/ /\ ~InSeq(order, k)
                /\ Len(order) = K
                /\ order' = Append(Tail(order), k)

Next == (\E k \in Keys : Get(k)) \/ (\E k \in Keys : Put(k))

Spec == Init /\ [][Next]_vars

\* Strong invariant: bounded length and all keys distinct.
Distinct(s) == \A i, j \in 1..Len(s) : i # j => s[i] # s[j]

Bounded == /\ Len(order) \in 0..K
           /\ Distinct(order)
           /\ \A i \in 1..Len(order) : order[i] \in Keys

TypeOK == /\ order \in Seq(Keys)
          /\ Bounded
====
