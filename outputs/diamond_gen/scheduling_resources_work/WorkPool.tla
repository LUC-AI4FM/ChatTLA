---- MODULE WorkPool ----
(***************************************************************************)
(* Worker pool of W threads pulling tasks from a bounded shared queue.    *)
(*                                                                         *)
(* Tasks transition queued -> in_flight -> done.  At most W tasks may be  *)
(* in_flight simultaneously, modeling a fixed-size worker pool.           *)
(*                                                                         *)
(* Safety: at most W in-flight; in-flight tasks are not visible in the    *)
(* queue; total tasks (queued + in_flight + done) is conserved.           *)
(***************************************************************************)
EXTENDS Naturals, FiniteSets

CONSTANT N

ASSUME N \in 2..3

W == 2  \* worker pool size
Tasks == 0..(N-1)

VARIABLES queued, inflight, done

vars == << queued, inflight, done >>

Init == /\ queued   = Tasks
        /\ inflight = {}
        /\ done     = {}

\* Worker pulls a task from the queue (if a worker slot is free).
Pull == /\ Cardinality(inflight) < W
        /\ \E t \in queued :
             /\ queued'   = queued \ {t}
             /\ inflight' = inflight \cup {t}
             /\ UNCHANGED done

\* Worker finishes the task it was processing.
Finish == /\ \E t \in inflight :
              /\ inflight' = inflight \ {t}
              /\ done'     = done \cup {t}
              /\ UNCHANGED queued

\* Recycle a finished task back into the queue (models a fresh job arriving
\* with the same id).  Keeps the state graph deadlock-free.
Recycle == /\ done # {}
           /\ \E t \in done :
                /\ done'   = done \ {t}
                /\ queued' = queued \cup {t}
                /\ UNCHANGED inflight

Next == Pull \/ Finish \/ Recycle

Spec == Init /\ [][Next]_vars

\* Strong safety: pool capacity, disjointness, conservation.
PoolInv ==
  /\ Cardinality(inflight) <= W
  /\ queued \cap inflight = {}
  /\ queued \cap done     = {}
  /\ inflight \cap done   = {}
  /\ (queued \cup inflight \cup done) = Tasks

TypeOK == /\ queued \subseteq Tasks /\ inflight \subseteq Tasks /\ done \subseteq Tasks /\ PoolInv
====
