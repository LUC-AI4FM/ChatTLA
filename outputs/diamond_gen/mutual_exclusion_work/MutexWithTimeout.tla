---- MODULE MutexWithTimeout ----
(***************************************************************************)
(* Mutex with a TryLock that gives up after a bounded number of attempts.  *)
(* If the timeout fires, the caller learns that someone else held the      *)
(* lock at that moment.  Mutual exclusion still holds: no two processes    *)
(* are simultaneously in the critical section.                             *)
(***************************************************************************)
EXTENDS Naturals

N == 2
Procs == 1..N
MaxAttempts == 2

VARIABLES pc, lock, attempts

vars == << pc, lock, attempts >>

NoHolder == 0

Init == /\ pc       = [i \in Procs |-> "ncs"]
        /\ lock     = NoHolder
        /\ attempts = [i \in Procs |-> 0]

\* Begin trying to acquire.
StartTry(i) ==
    /\ pc[i] = "ncs"
    /\ pc' = [pc EXCEPT ![i] = "trying"]
    /\ attempts' = [attempts EXCEPT ![i] = 0]
    /\ UNCHANGED lock

\* Successful TryLock.
Acquire(i) ==
    /\ pc[i] = "trying"
    /\ lock = NoHolder
    /\ lock' = i
    /\ pc' = [pc EXCEPT ![i] = "cs"]
    /\ UNCHANGED attempts

\* Failed attempt: bump counter.
Spin(i) ==
    /\ pc[i] = "trying"
    /\ lock # NoHolder
    /\ attempts[i] < MaxAttempts
    /\ attempts' = [attempts EXCEPT ![i] = attempts[i] + 1]
    /\ UNCHANGED << pc, lock >>

\* Timeout: give up after MaxAttempts.  Records that the lock was held
\* by someone else at the moment of timeout.
Timeout(i) ==
    /\ pc[i] = "trying"
    /\ lock # NoHolder
    /\ attempts[i] >= MaxAttempts
    /\ pc' = [pc EXCEPT ![i] = "timed_out"]
    /\ UNCHANGED << lock, attempts >>

\* Reset after a timeout.
Reset(i) ==
    /\ pc[i] = "timed_out"
    /\ pc' = [pc EXCEPT ![i] = "ncs"]
    /\ attempts' = [attempts EXCEPT ![i] = 0]
    /\ UNCHANGED lock

Release(i) ==
    /\ pc[i] = "cs"
    /\ lock = i
    /\ lock' = NoHolder
    /\ pc' = [pc EXCEPT ![i] = "ncs"]
    /\ UNCHANGED attempts

Idle == UNCHANGED vars

Next == \/ \E i \in Procs :
              StartTry(i) \/ Acquire(i) \/ Spin(i)
           \/ Timeout(i) \/ Reset(i) \/ Release(i)
        \/ Idle

Spec == Init /\ [][Next]_vars

TypeOK ==
    /\ pc       \in [Procs -> {"ncs","trying","cs","timed_out"}]
    /\ lock     \in Procs \cup {NoHolder}
    /\ attempts \in [Procs -> 0..MaxAttempts]
    /\ \A i, j \in Procs : (i # j /\ pc[i] = "cs") => pc[j] # "cs"
    /\ (lock # NoHolder) => pc[lock] = "cs"
====
