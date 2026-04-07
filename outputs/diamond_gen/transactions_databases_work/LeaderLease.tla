---- MODULE LeaderLease ----
(***************************************************************************)
(*  Leader lease for safe local reads.  At most one node may hold an    *)
(*  unexpired lease at any logical time.  A new leader can only be      *)
(*  elected after the previous leader's lease has expired.              *)
(***************************************************************************)
EXTENDS Naturals, FiniteSets

CONSTANTS Nodes, MaxTime, LeaseLen

VARIABLES leader, leaseEnd, now

vars == << leader, leaseEnd, now >>

NONE == "none"

Init == /\ leader   = NONE
        /\ leaseEnd = 0
        /\ now      = 0

\* Become leader: only allowed if no current leader OR the previous
\* lease has expired.
Elect(n) ==
    /\ \/ leader = NONE
       \/ now >= leaseEnd
    /\ leader'   = n
    /\ leaseEnd' = now + LeaseLen
    /\ UNCHANGED now

\* The leader serves a local read while inside its lease window.
LocalRead(n) ==
    /\ leader = n
    /\ now < leaseEnd
    /\ UNCHANGED vars

Tick == /\ now < MaxTime
        /\ now' = now + 1
        /\ UNCHANGED << leader, leaseEnd >>

Next == \/ \E n \in Nodes : Elect(n)
        \/ \E n \in Nodes : LocalRead(n)
        \/ Tick

Spec == Init /\ [][Next]_vars /\ WF_vars(Next)

\* Strong invariant: the leader -- if any -- has lease window in the
\* future of its election (leaseEnd > 0 once held).
TypeOK == /\ leader   \in Nodes \cup {NONE}
          /\ leaseEnd \in 0..(MaxTime + LeaseLen)
          /\ now      \in 0..MaxTime
          /\ (leader # NONE) => (leaseEnd >= LeaseLen)
          /\ (leader = NONE) => (leaseEnd = 0 \/ leaseEnd <= now)
====
