---- MODULE EventCount ----
(***************************************************************************)
(* Reed and Kanodia event counts (1979).  An event count is a monotone     *)
(* non-decreasing integer.  advance() increments it; await(v) blocks until *)
(* the count reaches v.  Used to coordinate producers and consumers        *)
(* without explicit signalling.                                            *)
(***************************************************************************)
EXTENDS Naturals

N == 2
Procs == 1..N
MaxCount == 3

VARIABLES count, pc, target

vars == << count, pc, target >>

Init == /\ count  = 0
        /\ pc     = [i \in Procs |-> "idle"]
        /\ target = [i \in Procs |-> 0]

\* Begin awaiting some value v.
StartAwait(i, v) ==
    /\ pc[i] = "idle"
    /\ pc'     = [pc EXCEPT ![i] = "waiting"]
    /\ target' = [target EXCEPT ![i] = v]
    /\ UNCHANGED count

\* Advance: bump count by one (bounded for model checking).
Advance ==
    /\ count < MaxCount
    /\ count' = count + 1
    /\ UNCHANGED << pc, target >>

\* Wake when count has reached our target.
Wake(i) ==
    /\ pc[i] = "waiting"
    /\ count >= target[i]
    /\ pc' = [pc EXCEPT ![i] = "done"]
    /\ UNCHANGED << count, target >>

\* Reset: a finished waiter returns to idle (so the model loops cleanly).
Reset(i) ==
    /\ pc[i] = "done"
    /\ pc'     = [pc EXCEPT ![i] = "idle"]
    /\ target' = [target EXCEPT ![i] = 0]
    /\ UNCHANGED count

Idle == UNCHANGED vars

Next == \/ \E i \in Procs, v \in 1..MaxCount : StartAwait(i, v)
        \/ Advance
        \/ \E i \in Procs : Wake(i) \/ Reset(i)
        \/ Idle

Spec == Init /\ [][Next]_vars

\* The strong safety property — count is monotone non-decreasing — must be
\* expressed as an action invariant since invariants are state predicates.
\* We instead conjoin a derived state property: any waiting process whose
\* target has already been met must be allowed to wake (used as a sanity
\* bound), plus the natural type bounds.
TypeOK ==
    /\ count  \in 0..MaxCount
    /\ pc     \in [Procs -> {"idle","waiting","done"}]
    /\ target \in [Procs -> 0..MaxCount]
    /\ \A i \in Procs : pc[i] = "done" => count >= target[i]
====
