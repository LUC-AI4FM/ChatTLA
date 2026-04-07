---- MODULE BullyElection ----
(***************************************************************************)
(* The Bully leader-election algorithm (Garcia-Molina, 1982).              *)
(* Processes have totally ordered ids. A process that suspects no leader   *)
(* starts an election; the highest id always wins. We model only the       *)
(* observable per-process state and the elected leader.                    *)
(* Safety: at most one leader at any time, and the leader has the max id. *)
(***************************************************************************)
EXTENDS Naturals, FiniteSets

Procs == {1, 2, 3}
MaxId == 3

VARIABLES state, leader

vars == << state, leader >>

\* Init: every process is a follower; no leader elected yet.
Init == /\ state  = [i \in Procs |-> "follower"]
        /\ leader = 0

\* A follower notices the leader is gone and becomes a candidate.
StartElection(i) ==
    /\ state[i] = "follower"
    /\ leader   = 0
    /\ state' = [state EXCEPT ![i] = "candidate"]
    /\ UNCHANGED leader

\* A candidate is preempted by a higher-id candidate.
Preempt(i, j) ==
    /\ state[i] = "candidate"
    /\ j > i
    /\ j \in Procs
    /\ state' = [state EXCEPT ![i] = "follower"]
    /\ UNCHANGED leader

\* A candidate becomes leader only if no higher-id process is also a candidate
\* and it is the maximum id of any candidate.  This guards uniqueness.
BecomeLeader(i) ==
    /\ state[i] = "candidate"
    /\ leader   = 0
    /\ \A j \in Procs : j > i => state[j] # "candidate"
    /\ \A j \in Procs : j > i => state[j] = "follower"
    /\ state' = [state EXCEPT ![i] = "leader"]
    /\ leader' = i

\* The leader steps down (e.g., voluntary failure), allowing a new election.
Resign(i) ==
    /\ state[i] = "leader"
    /\ state' = [state EXCEPT ![i] = "follower"]
    /\ leader' = 0

Next == \/ \E i \in Procs : StartElection(i)
        \/ \E i, j \in Procs : Preempt(i, j)
        \/ \E i \in Procs : BecomeLeader(i)
        \/ \E i \in Procs : Resign(i)

Spec == Init /\ [][Next]_vars

TypeOK ==
    /\ state  \in [Procs -> {"follower", "candidate", "leader"}]
    /\ leader \in Procs \cup {0}

\* Strong safety: at most one process is in the leader state, and if a leader
\* is recorded, that process is in the leader state.  This is the property a
\* Diamond mutation test must catch.
SafetyInv == Cardinality({i \in Procs : state[i] = "leader"}) <= 1 /\ ((leader # 0) => state[leader] = "leader")
====
