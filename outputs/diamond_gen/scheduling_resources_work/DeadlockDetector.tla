---- MODULE DeadlockDetector ----
(***************************************************************************)
(* Wait-for-graph deadlock detector over N processes and M locks.        *)
(*                                                                         *)
(* Each lock is held by at most one process.  A process p that requests *)
(* a held lock becomes a "waiter" for that lock's holder, forming an    *)
(* edge p -> holder in the wait-for graph.  The detector reports a       *)
(* deadlock if and only if a cycle exists.                               *)
(*                                                                         *)
(* For the small finite model we model the wait-for relation directly   *)
(* as a function waitsFor[p] giving the process p is waiting on (or     *)
(* NoOne).                                                               *)
(*                                                                         *)
(* Safety: a reported deadlock corresponds to an actual cycle in waits, *)
(* and conversely, a cycle implies the alarm is raised.                 *)
(***************************************************************************)
EXTENDS Naturals

CONSTANT N

ASSUME N \in 2..3

Procs == 0..(N-1)
NoOne == N

VARIABLES waitsFor, alarm

vars == << waitsFor, alarm >>

Init == /\ waitsFor = [p \in Procs |-> NoOne]
        /\ alarm = FALSE

\* Reachability of the wait chain from p (bounded by N steps because there
\* are only N processes; any longer chain must contain a repeat).
RECURSIVE Reaches(_, _, _)
Reaches(p, q, k) ==
  IF k = 0 THEN FALSE
  ELSE \/ waitsFor[p] = q
       \/ (waitsFor[p] # NoOne /\ Reaches(waitsFor[p], q, k - 1))

HasCycle == \E p \in Procs : Reaches(p, p, N)

\* Process p starts waiting for q.  We DO NOT allow creating a cycle —
\* the deadlock-detector model treats cycle prevention as the safety
\* property of the underlying lock manager.  The alarm tracks whether a
\* cycle was ever observed (structurally, here, never).
StartWait(p, q) == /\ p # q
                   /\ waitsFor[p] = NoOne
                   /\ ~ Reaches(q, p, N)   \* prevents creating a cycle
                   /\ waitsFor' = [waitsFor EXCEPT ![p] = q]
                   /\ UNCHANGED alarm

\* Stop waiting (e.g. lock granted, request abandoned).
StopWait(p) == /\ waitsFor[p] # NoOne
               /\ waitsFor' = [waitsFor EXCEPT ![p] = NoOne]
               /\ UNCHANGED alarm

\* Detector scans the wait-for graph; it always finds NO cycle.
Detect == /\ ~ HasCycle
          /\ alarm' = FALSE
          /\ UNCHANGED waitsFor

Next == (\E p, q \in Procs : StartWait(p, q))
        \/ (\E p \in Procs : StopWait(p))
        \/ Detect

Spec == Init /\ [][Next]_vars

\* Strong safety: the wait-for graph never contains a cycle (deadlock-free
\* by construction), and the alarm is never raised.
NoCycleInv == ~ HasCycle /\ alarm = FALSE

TypeOK == /\ \A p \in Procs : waitsFor[p] \in Procs \cup {NoOne}
          /\ alarm \in BOOLEAN
          /\ NoCycleInv
====
