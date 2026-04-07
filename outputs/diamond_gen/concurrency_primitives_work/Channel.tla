---- MODULE Channel ----
EXTENDS Naturals, Sequences

CONSTANTS K, MaxVal

\* buffer is a sequence of values; capacity K.
VARIABLE buffer

vars == << buffer >>

Init == buffer = << >>

\* Send: blocks when buffer is full.
Send == /\ Len(buffer) < K
        /\ \E v \in 1..MaxVal :
              buffer' = Append(buffer, v)

\* Recv: blocks when buffer is empty.
Recv == /\ Len(buffer) > 0
        /\ buffer' = Tail(buffer)

Next == \/ Send
        \/ Recv

Spec == Init /\ [][Next]_vars

\* Safety: 0 <= len(buffer) <= K and all values are valid.
ChannelSafe == /\ Len(buffer) <= K
               /\ \A i \in 1..Len(buffer) : buffer[i] \in 1..MaxVal

TypeOK == /\ buffer \in Seq(1..MaxVal)
          /\ Len(buffer) <= K
          /\ ChannelSafe
====
