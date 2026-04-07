---- MODULE AravindMutex ----
(***************************************************************************)
(* Aravind's local-spin queue lock variant.  Each process has a private    *)
(* boolean wait flag.  When the previous holder releases, it clears the    *)
(* wait flag of its successor (FIFO order).  This bounds bypass to N-1.    *)
(***************************************************************************)
EXTENDS Naturals, Sequences

N == 3
Procs == 1..N

VARIABLES pc, waiting, queue

vars == << pc, waiting, queue >>

\* `queue` is a sequence of process ids in arrival order.

Init == /\ pc      = [i \in Procs |-> "ncs"]
        /\ waiting = [i \in Procs |-> TRUE]
        /\ queue   = << >>

\* Enqueue self when starting to acquire.
Enqueue(i) ==
    /\ pc[i] = "ncs"
    /\ \A k \in 1..Len(queue) : queue[k] # i  \* not already enqueued
    /\ queue' = Append(queue, i)
    /\ pc' = [pc EXCEPT ![i] = "wait"]
    /\ waiting' = IF queue = << >>
                    THEN [waiting EXCEPT ![i] = FALSE]
                    ELSE waiting

\* Spin until our private wait flag is cleared.
EnterCS(i) ==
    /\ pc[i] = "wait"
    /\ waiting[i] = FALSE
    /\ pc' = [pc EXCEPT ![i] = "cs"]
    /\ UNCHANGED << waiting, queue >>

\* Release: pop self from head, clear successor's wait flag.
Release(i) ==
    /\ pc[i] = "cs"
    /\ Len(queue) > 0
    /\ Head(queue) = i
    /\ LET tl == Tail(queue) IN
       /\ queue' = tl
       /\ waiting' =
            IF tl = << >>
              THEN [waiting EXCEPT ![i] = TRUE]
              ELSE [waiting EXCEPT ![i] = TRUE, ![Head(tl)] = FALSE]
    /\ pc' = [pc EXCEPT ![i] = "ncs"]

\* Bound the queue length to keep TLC's state space finite.
QueueBound == Len(queue) <= N

Idle == UNCHANGED vars

Next == \/ \E i \in Procs : Enqueue(i) \/ EnterCS(i) \/ Release(i)
        \/ Idle

Spec == Init /\ [][Next]_vars

TypeOK ==
    /\ pc      \in [Procs -> {"ncs","wait","cs"}]
    /\ waiting \in [Procs -> BOOLEAN]
    /\ queue   \in Seq(Procs)
    /\ Len(queue) <= N
    /\ \A i, j \in Procs : (i # j /\ pc[i] = "cs") => pc[j] # "cs"
====
