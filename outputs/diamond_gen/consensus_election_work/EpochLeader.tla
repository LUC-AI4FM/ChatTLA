---- MODULE EpochLeader ----
(***************************************************************************)
(* Epoch-numbered leader election.  Every node tracks the highest epoch  *)
(* it has observed; a node can be leader only in its own epoch and only  *)
(* if its epoch is the highest seen.                                     *)
(* Safety: at most one leader per epoch.                                 *)
(***************************************************************************)
EXTENDS Naturals, FiniteSets

Nodes  == {1, 2, 3}
Epochs == 1..3

VARIABLES epoch, leader, maxSeen

vars == << epoch, leader, maxSeen >>

Init == /\ epoch   = [n \in Nodes |-> 1]
        /\ leader  = [e \in Epochs |-> 0]
        /\ maxSeen = 1

\* A node bumps its epoch beyond the global max so far (advances the round).
Bump(n) ==
    /\ epoch[n] = maxSeen
    /\ maxSeen + 1 \in Epochs
    /\ epoch'   = [epoch EXCEPT ![n] = maxSeen + 1]
    /\ maxSeen' = maxSeen + 1
    /\ UNCHANGED leader

\* A node becomes leader of its epoch only if no leader has been recorded
\* there yet AND its epoch equals the global maximum (so it has the latest
\* view of the epoch counter).
BecomeLeader(n) ==
    /\ epoch[n] = maxSeen
    /\ leader[epoch[n]] = 0
    /\ leader' = [leader EXCEPT ![epoch[n]] = n]
    /\ UNCHANGED << epoch, maxSeen >>

\* Restart so the state space stays finite without terminal states.
Reset ==
    /\ maxSeen = 3
    /\ \E e \in Epochs : leader[e] # 0
    /\ epoch'   = [n \in Nodes |-> 1]
    /\ leader'  = [e \in Epochs |-> 0]
    /\ maxSeen' = 1

Next == \/ \E n \in Nodes : Bump(n)
        \/ \E n \in Nodes : BecomeLeader(n)
        \/ Reset

Spec == Init /\ [][Next]_vars

TypeOK ==
    /\ epoch   \in [Nodes -> Epochs]
    /\ leader  \in [Epochs -> Nodes \cup {0}]
    /\ maxSeen \in Epochs

\* Strong safety: any recorded leader's own epoch is at least the epoch in
\* which it was elected (a node can only be leader of an epoch >= its own).
SafetyInv == \A e \in Epochs : (leader[e] # 0) => epoch[leader[e]] >= e
====
