---- MODULE ChangRoberts ----
(***************************************************************************)
(* Chang-Roberts ring leader election (1979).                              *)
(* Processes are arranged in a unidirectional ring with unique ids.        *)
(* Each process forwards a token id to its successor only if the id is     *)
(* greater than the process's own id; otherwise the token is suppressed.   *)
(* The token whose id equals its current holder identifies the leader.    *)
(* Safety: at most one leader, and the leader has the maximum id.          *)
(***************************************************************************)
EXTENDS Naturals, FiniteSets

N == 3
Procs == 1..N

Succ(i) == (i % N) + 1

\* msgs is the multiset of in-flight tokens, modeled as a set of records
\* << dest, val >>.  Initially each process i sends its own id to Succ(i).
VARIABLES msgs, leader

vars == << msgs, leader >>

Msg == [dest : Procs, val : Procs]

Init == /\ msgs   = { [dest |-> Succ(i), val |-> i] : i \in Procs }
        /\ leader = 0

\* A process forwards a strictly larger id (suppresses smaller ones).
\* Only happens before election.
Forward(m) ==
    /\ m \in msgs
    /\ m.val > m.dest
    /\ leader = 0
    /\ msgs' = (msgs \ {m}) \cup { [dest |-> Succ(m.dest), val |-> m.val] }
    /\ UNCHANGED leader

\* A process drops a token whose id is smaller than its own id (suppression),
\* or drains any leftover token after the leader is known.
Drop(m) ==
    /\ m \in msgs
    /\ \/ m.val < m.dest
       \/ leader # 0
    /\ msgs' = msgs \ {m}
    /\ UNCHANGED leader

\* A process whose own id has come back to it declares itself leader.
Elect(m) ==
    /\ m \in msgs
    /\ m.val = m.dest
    /\ leader = 0
    /\ leader' = m.dest
    /\ msgs'   = msgs \ {m}

\* Reset the ring after an election so the protocol can run again.  This
\* keeps the state space finite while avoiding terminal deadlock states.
Reset ==
    /\ leader # 0
    /\ msgs   = {}
    /\ msgs'  = { [dest |-> Succ(i), val |-> i] : i \in Procs }
    /\ leader' = 0

Next == \/ \E m \in msgs : Forward(m)
        \/ \E m \in msgs : Drop(m)
        \/ \E m \in msgs : Elect(m)
        \/ Reset

Spec == Init /\ [][Next]_vars

TypeOK ==
    /\ msgs   \subseteq Msg
    /\ leader \in Procs \cup {0}

\* Strong safety: only the maximum id can ever be elected leader.
SafetyInv == (leader # 0) => (\A j \in Procs : j <= leader)
====
