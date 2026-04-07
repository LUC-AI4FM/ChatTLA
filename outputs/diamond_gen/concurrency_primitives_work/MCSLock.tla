---- MODULE MCSLock ----
EXTENDS Naturals, Sequences, FiniteSets

CONSTANT Procs

\* queue : a sequence of process ids waiting for the lock; head is the holder.
VARIABLE queue

vars == << queue >>

\* No duplicates in the queue.
NoDup(s) == \A i, j \in 1..Len(s) : i # j => s[i] # s[j]

InQueue(p) == \E i \in 1..Len(queue) : queue[i] = p

Init == queue = << >>

\* Enqueue: add self to the tail of the queue.
Enqueue(p) == /\ ~InQueue(p)
              /\ queue' = Append(queue, p)

\* Release: only the head of the queue may release; remove it.
Release(p) == /\ Len(queue) > 0
              /\ queue[1] = p
              /\ queue' = Tail(queue)

Next == \/ \E p \in Procs : Enqueue(p)
        \/ \E p \in Procs : Release(p)

Spec == Init /\ [][Next]_vars

\* Strong safety: queue length is bounded, no duplicates, head is unique holder.
MCSSafe == /\ Len(queue) <= Cardinality(Procs)
           /\ NoDup(queue)

TypeOK == /\ queue \in Seq(Procs)
          /\ Len(queue) <= Cardinality(Procs)
          /\ MCSSafe
====
