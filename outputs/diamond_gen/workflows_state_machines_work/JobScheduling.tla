---- MODULE JobScheduling ----
(***************************************************************************)
(*  Job scheduler with bounded retries.                                    *)
(*    queued -> running -> succeeded                                       *)
(*    queued -> running -> failed -> queued (until MaxRetries reached)     *)
(***************************************************************************)
EXTENDS Naturals

CONSTANT MaxRetries

VARIABLES status, retries, ever_run

vars == << status, retries, ever_run >>

States == {"queued", "running", "succeeded", "failed_terminal"}

Init == /\ status   = "queued"
        /\ retries  = 0
        /\ ever_run = FALSE

Start == /\ status = "queued"
         /\ status' = "running"
         /\ ever_run' = TRUE
         /\ UNCHANGED retries

Succeed == /\ status = "running"
           /\ status' = "succeeded"
           /\ UNCHANGED << retries, ever_run >>

Fail == /\ status = "running"
        /\ retries < MaxRetries
        /\ retries' = retries + 1
        /\ status' = "queued"
        /\ UNCHANGED ever_run

Abandon == /\ status = "running"
           /\ retries = MaxRetries
           /\ status' = "failed_terminal"
           /\ UNCHANGED << retries, ever_run >>

Done == /\ status \in {"succeeded", "failed_terminal"}
        /\ UNCHANGED vars

Next == \/ Start \/ Succeed \/ Fail \/ Abandon \/ Done

Spec == Init /\ [][Next]_vars

\* Succeeded is terminal and required at least one run; retries are bounded.
SafetyInvariant == (retries <= MaxRetries) /\ ((status = "succeeded") => ever_run) /\ ((status = "failed_terminal") => (ever_run /\ retries = MaxRetries))

TypeOK == /\ status \in States
          /\ retries \in 0..MaxRetries
          /\ ever_run \in BOOLEAN
          /\ SafetyInvariant
====
