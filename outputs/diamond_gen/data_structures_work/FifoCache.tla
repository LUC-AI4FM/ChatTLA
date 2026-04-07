---- MODULE FifoCache ----
EXTENDS Naturals, Sequences

CONSTANTS K, Keys

VARIABLES cache  \* sequence of distinct keys in insertion order; head = oldest

vars == << cache >>

InSeq(s, k) == \E i \in 1..Len(s) : s[i] = k

Init == cache = << >>

\* Insert: if absent, append; if full, evict oldest first.
Insert(k) == /\ k \in Keys
             /\ ~InSeq(cache, k)
             /\ \/ /\ Len(cache) < K
                   /\ cache' = Append(cache, k)
                \/ /\ Len(cache) = K
                   /\ cache' = Append(Tail(cache), k)

\* Explicit removal of an arbitrary present key.
Remove(k) == /\ InSeq(cache, k)
             /\ LET idx == CHOOSE i \in 1..Len(cache) : cache[i] = k
                IN cache' = SubSeq(cache, 1, idx - 1) \o SubSeq(cache, idx + 1, Len(cache))

Next == (\E k \in Keys : Insert(k)) \/ (\E k \in Keys : Remove(k))

Spec == Init /\ [][Next]_vars

Distinct(s) == \A i, j \in 1..Len(s) : i # j => s[i] # s[j]

Bounded == /\ Len(cache) \in 0..K
           /\ Distinct(cache)
           /\ \A i \in 1..Len(cache) : cache[i] \in Keys

TypeOK == /\ cache \in Seq(Keys)
          /\ Bounded
====
