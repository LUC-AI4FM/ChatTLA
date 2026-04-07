---- MODULE Once ----
EXTENDS Naturals, FiniteSets

CONSTANT Procs

\* state in {0=pending, 1=running, 2=done}
\* runs : how many times the action has actually executed (must stay <= 1)
VARIABLES state, runs, callers

vars == << state, runs, callers >>

Pending == 0
Running == 1
Done    == 2

Init == /\ state = Pending
        /\ runs  = 0
        /\ callers = {}

\* The first caller transitions pending -> running and runs the action.
StartFirst(p) == /\ state = Pending
                 /\ p \notin callers
                 /\ state' = Running
                 /\ runs'  = runs + 1
                 /\ callers' = callers \cup {p}

\* Action completes; state becomes done.
Finish == /\ state = Running
          /\ state' = Done
          /\ UNCHANGED << runs, callers >>

\* Subsequent callers: they observe done state without re-running.
LateCall(p) == /\ state = Done
               /\ p \notin callers
               /\ callers' = callers \cup {p}
               /\ UNCHANGED << state, runs >>

\* Concurrent caller arriving while running: just waits (records itself).
WaitCall(p) == /\ state = Running
               /\ p \notin callers
               /\ callers' = callers \cup {p}
               /\ UNCHANGED << state, runs >>

\* "Restart": once everybody has observed Done, they all leave the Once region.
\* This avoids terminal deadlock; it does NOT re-arm the action (runs stays = 1).
LeaveAll == /\ state = Done
            /\ callers = Procs
            /\ callers' = {}
            /\ UNCHANGED << state, runs >>

Next == \/ \E p \in Procs : StartFirst(p)
        \/ Finish
        \/ \E p \in Procs : LateCall(p)
        \/ \E p \in Procs : WaitCall(p)
        \/ LeaveAll

Spec == Init /\ [][Next]_vars

\* Safety: the action runs AT MOST ONCE.
OnceSafe == /\ runs <= 1
            /\ ((state = Pending) => (runs = 0))
            /\ ((state \in {Running, Done}) => (runs = 1))

TypeOK == /\ state \in {Pending, Running, Done}
          /\ runs  \in 0..1
          /\ callers \subseteq Procs
          /\ OnceSafe
====
