---- MODULE AdaptiveMutex ----
(***************************************************************************)
(* Adaptive mutex (Solaris-style): a thread first spins for a bounded      *)
(* number of cycles; if it cannot acquire it transitions to a blocking     *)
(* sleep.  Per-thread mode is one of {spinning, sleeping, holding}.        *)
(***************************************************************************)
EXTENDS Naturals

N == 2
Procs == 1..N
MaxSpin == 2
NoHolder == 0

VARIABLES mode, lock, spinCount

vars == << mode, lock, spinCount >>

Init == /\ mode      = [i \in Procs |-> "idle"]
        /\ lock      = NoHolder
        /\ spinCount = [i \in Procs |-> 0]

\* Begin spinning.
StartSpin(i) ==
    /\ mode[i] = "idle"
    /\ mode' = [mode EXCEPT ![i] = "spinning"]
    /\ spinCount' = [spinCount EXCEPT ![i] = 0]
    /\ UNCHANGED lock

\* Spin succeeds: acquire the lock.
SpinAcquire(i) ==
    /\ mode[i] = "spinning"
    /\ lock = NoHolder
    /\ lock' = i
    /\ mode' = [mode EXCEPT ![i] = "holding"]
    /\ UNCHANGED spinCount

\* Spin step that fails to acquire: bump counter.
SpinStep(i) ==
    /\ mode[i] = "spinning"
    /\ lock # NoHolder
    /\ spinCount[i] < MaxSpin
    /\ spinCount' = [spinCount EXCEPT ![i] = spinCount[i] + 1]
    /\ UNCHANGED << mode, lock >>

\* Spin budget exhausted: go to sleep.
GoToSleep(i) ==
    /\ mode[i] = "spinning"
    /\ lock # NoHolder
    /\ spinCount[i] >= MaxSpin
    /\ mode' = [mode EXCEPT ![i] = "sleeping"]
    /\ UNCHANGED << lock, spinCount >>

\* Sleeping thread is woken when the lock becomes free.
WakeAndAcquire(i) ==
    /\ mode[i] = "sleeping"
    /\ lock = NoHolder
    /\ lock' = i
    /\ mode' = [mode EXCEPT ![i] = "holding"]
    /\ spinCount' = [spinCount EXCEPT ![i] = 0]

Release(i) ==
    /\ mode[i] = "holding"
    /\ lock = i
    /\ lock' = NoHolder
    /\ mode' = [mode EXCEPT ![i] = "idle"]
    /\ UNCHANGED spinCount

Idle == UNCHANGED vars

Next == \/ \E i \in Procs :
              StartSpin(i) \/ SpinAcquire(i) \/ SpinStep(i)
           \/ GoToSleep(i) \/ WakeAndAcquire(i) \/ Release(i)
        \/ Idle

Spec == Init /\ [][Next]_vars

\* Strong safety: at most one process is in the holding (CS) mode.
Mutex == \A i, j \in Procs : (i # j /\ mode[i] = "holding") => mode[j] # "holding"

TypeOK ==
    /\ mode      \in [Procs -> {"idle","spinning","sleeping","holding"}]
    /\ lock      \in Procs \cup {NoHolder}
    /\ spinCount \in [Procs -> 0..MaxSpin]
    /\ (lock # NoHolder) => mode[lock] = "holding"
    /\ Mutex
====
