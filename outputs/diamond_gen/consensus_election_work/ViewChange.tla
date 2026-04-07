---- MODULE ViewChange ----
(***************************************************************************)
(* View-change protocol (PBFT-style abstraction).  Each replica may       *)
(* propose a new view number; once a quorum of view-change messages for a *)
(* given view has accumulated, that view installs and its primary takes  *)
(* over.  Safety: at most one primary per view.                          *)
(***************************************************************************)
EXTENDS Naturals, FiniteSets

Replicas == {1, 2, 3}
Views    == 1..2
Quorum   == {Q \in SUBSET Replicas : Cardinality(Q) >= 2}

\* Primary of a view is determined deterministically by view number.
PrimaryOf(v) == ((v - 1) % Cardinality(Replicas)) + 1

VARIABLES vcMsgs, currentView, primary

vars == << vcMsgs, currentView, primary >>

Init == /\ vcMsgs      = [v \in Views |-> {}]
        /\ currentView = [r \in Replicas |-> 1]
        /\ primary     = [v \in Views |-> 0]

\* Replica r proposes view v (must be strictly greater than its current view).
ProposeView(r, v) ==
    /\ v > currentView[r]
    /\ r \notin vcMsgs[v]
    /\ vcMsgs'      = [vcMsgs EXCEPT ![v] = @ \cup {r}]
    /\ currentView' = [currentView EXCEPT ![r] = v]
    /\ UNCHANGED primary

\* Once a quorum of replicas has signed a view-change for view v, the new
\* primary takes over (only one such installation per view).
Install(v) ==
    /\ vcMsgs[v] \in Quorum
    /\ primary[v] = 0
    /\ primary' = [primary EXCEPT ![v] = PrimaryOf(v)]
    /\ UNCHANGED << vcMsgs, currentView >>

\* Restart so the state space stays finite.
Reset ==
    /\ \E v \in Views : primary[v] # 0
    /\ vcMsgs'      = [v \in Views |-> {}]
    /\ currentView' = [r \in Replicas |-> 1]
    /\ primary'     = [v \in Views |-> 0]

Next == \/ \E r \in Replicas : \E v \in Views : ProposeView(r, v)
        \/ \E v \in Views : Install(v)
        \/ Reset

Spec == Init /\ [][Next]_vars

TypeOK ==
    /\ vcMsgs      \in [Views -> SUBSET Replicas]
    /\ currentView \in [Replicas -> Views]
    /\ primary     \in [Views -> Replicas \cup {0}]

\* Strong safety: at most one primary per view (and that primary, if any,
\* matches the deterministic primary-of-view function).
SafetyInv == \A v \in Views : (primary[v] # 0) => primary[v] = PrimaryOf(v)
====
