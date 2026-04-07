---- MODULE QueuedRequests ----
(***************************************************************************)
(*  Bounded request queue with strict FIFO commit.  Requests are        *)
(*  enqueued at the tail and committed from the head.                   *)
(*                                                                         *)
(*  Strong invariant: the commit log is exactly the prefix of the       *)
(*  enqueued sequence; commit order matches enqueue order.              *)
(***************************************************************************)
EXTENDS Naturals, Sequences

CONSTANTS MaxQueue

VARIABLES enq, queue, committed

vars == << enq, queue, committed >>

Init == /\ enq       = << >>
        /\ queue     = << >>
        /\ committed = << >>

\* Enqueue: append a fresh request id (= length+1) to both the
\* permanent enqueue history and the live queue.
Enqueue == /\ Len(enq) < MaxQueue
           /\ enq'   = Append(enq,   Len(enq) + 1)
           /\ queue' = Append(queue, Len(enq) + 1)
           /\ UNCHANGED committed

\* Commit the head of the live queue.
Commit == /\ Len(queue) > 0
          /\ committed' = Append(committed, Head(queue))
          /\ queue'     = Tail(queue)
          /\ UNCHANGED enq

Next == \/ Enqueue \/ Commit

Spec == Init /\ [][Next]_vars /\ WF_vars(Next)

\* Strong invariant: committed is a prefix of enq, queue is the suffix.
TypeOK == /\ enq       \in Seq(1..MaxQueue)
          /\ queue     \in Seq(1..MaxQueue)
          /\ committed \in Seq(1..MaxQueue)
          /\ Len(enq) <= MaxQueue
          /\ Len(committed) + Len(queue) = Len(enq)
          /\ \A i \in 1..Len(committed) : committed[i] = enq[i]
          /\ \A i \in 1..Len(queue) : queue[i] = enq[Len(committed) + i]
====
