---- MODULE RecursiveMutex ----
(***************************************************************************)
(* A recursive (re-entrant) mutex.  The same thread may acquire the lock   *)
(* multiple times; a depth counter tracks the recursion.  Only the         *)
(* current holder may decrement, and the lock is freed only when depth     *)
(* drops to zero.  Mutual exclusion still holds: at most one holder.       *)
(***************************************************************************)
EXTENDS Naturals

N == 2
Procs == 1..N
MaxDepth == 2
NoHolder == 0

VARIABLES holder, depth, pc

vars == << holder, depth, pc >>

Init == /\ holder = NoHolder
        /\ depth  = 0
        /\ pc     = [i \in Procs |-> "ncs"]

\* Acquire when free.
AcquireFresh(i) ==
    /\ pc[i] = "ncs"
    /\ holder = NoHolder
    /\ holder' = i
    /\ depth'  = 1
    /\ pc'     = [pc EXCEPT ![i] = "cs"]

\* Re-acquire (recursive).
AcquireRecursive(i) ==
    /\ pc[i] = "cs"
    /\ holder = i
    /\ depth < MaxDepth
    /\ depth' = depth + 1
    /\ UNCHANGED << holder, pc >>

\* Release one level.
ReleaseOne(i) ==
    /\ pc[i] = "cs"
    /\ holder = i
    /\ depth > 1
    /\ depth' = depth - 1
    /\ UNCHANGED << holder, pc >>

\* Release final level.
ReleaseFinal(i) ==
    /\ pc[i] = "cs"
    /\ holder = i
    /\ depth = 1
    /\ holder' = NoHolder
    /\ depth'  = 0
    /\ pc'     = [pc EXCEPT ![i] = "ncs"]

Idle == UNCHANGED vars

Next == \/ \E i \in Procs :
              AcquireFresh(i) \/ AcquireRecursive(i)
           \/ ReleaseOne(i) \/ ReleaseFinal(i)
        \/ Idle

Spec == Init /\ [][Next]_vars

TypeOK ==
    /\ holder \in Procs \cup {NoHolder}
    /\ depth  \in 0..MaxDepth
    /\ pc     \in [Procs -> {"ncs","cs"}]
    /\ (holder = NoHolder) <=> (depth = 0)
    /\ (holder # NoHolder) => pc[holder] = "cs"
    /\ \A i, j \in Procs : (i # j /\ pc[i] = "cs") => pc[j] # "cs"
====
