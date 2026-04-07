---- MODULE RetryWithBackoff ----
(***************************************************************************)
(* Bounded exponential backoff retry policy.  Each failure doubles the    *)
(* current delay (1, 2, 4, ...) up to MaxDelay.  After MaxAttempts the    *)
(* operation gives up.                                                     *)
(* Strong safety: attempts <= MaxAttempts and delay is always a power of  *)
(* two within {1, 2, 4, ..., MaxDelay}.                                    *)
(***************************************************************************)
EXTENDS Naturals

MaxAttempts == 4
MaxDelay == 8

VARIABLES attempts, delay, status

vars == << attempts, delay, status >>

\* status in {"running", "succeeded", "given_up"}.
Init ==
    /\ attempts = 0
    /\ delay = 1
    /\ status = "running"

\* A successful attempt: clears the retry loop.
Succeed ==
    /\ status = "running"
    /\ attempts < MaxAttempts
    /\ status' = "succeeded"
    /\ attempts' = attempts + 1
    /\ UNCHANGED delay

\* A failure: double the delay (capped) and retry.
FailAndBackoff ==
    /\ status = "running"
    /\ attempts < MaxAttempts - 1
    /\ attempts' = attempts + 1
    /\ delay' = IF delay * 2 <= MaxDelay THEN delay * 2 ELSE MaxDelay
    /\ UNCHANGED status

\* Final failure exhausts the retry budget.
GiveUp ==
    /\ status = "running"
    /\ attempts = MaxAttempts - 1
    /\ attempts' = attempts + 1
    /\ status' = "given_up"
    /\ UNCHANGED delay

Done == UNCHANGED vars

Next == Succeed \/ FailAndBackoff \/ GiveUp \/ Done

Spec == Init /\ [][Next]_vars

\* Strong safety conjoined into TypeOK: attempts bounded; delay always
\* a power of two in {1,2,4,8}; success/give_up are terminal.
PowOf2(d) == d \in {1, 2, 4, 8}

TypeOK ==
    /\ attempts \in 0 .. MaxAttempts
    /\ delay \in 1 .. MaxDelay
    /\ status \in {"running", "succeeded", "given_up"}
    /\ PowOf2(delay)
    /\ delay <= MaxDelay
    /\ (status = "succeeded") => (attempts >= 1)
    /\ (status = "given_up") => (attempts = MaxAttempts)
====
