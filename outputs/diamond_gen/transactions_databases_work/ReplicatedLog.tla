---- MODULE ReplicatedLog ----
(***************************************************************************)
(*  A simple replicated append-only log over N replicas with majority    *)
(*  commit.  An entry is committed once a majority of replicas have a   *)
(*  copy of it.                                                          *)
(*                                                                         *)
(*  Strong invariant: every committed prefix is present on at least a   *)
(*  majority of replicas; therefore any two majorities intersect on it. *)
(***************************************************************************)
EXTENDS Naturals, FiniteSets

CONSTANTS Replicas, MaxLog

VARIABLES logLen, committed

vars == << logLen, committed >>

Quorum == (Cardinality(Replicas) \div 2) + 1

Init == /\ logLen    = [r \in Replicas |-> 0]
        /\ committed = 0

\* The leader (any replica) appends a new entry to its own log.
LeaderAppend(r) ==
    /\ logLen[r] < MaxLog
    /\ logLen' = [logLen EXCEPT ![r] = @ + 1]
    /\ UNCHANGED committed

\* A follower copies the next missing entry from another replica.
Replicate(src, dst) ==
    /\ src # dst
    /\ logLen[dst] < logLen[src]
    /\ logLen[dst] < MaxLog
    /\ logLen' = [logLen EXCEPT ![dst] = @ + 1]
    /\ UNCHANGED committed

\* Advance the commit index when a majority of replicas have at least N
\* entries.
Commit ==
    /\ \E n \in 1..MaxLog :
          /\ n > committed
          /\ Cardinality({r \in Replicas : logLen[r] >= n}) >= Quorum
          /\ committed' = n
    /\ UNCHANGED logLen

Next == \/ \E r \in Replicas : LeaderAppend(r)
        \/ \E s, d \in Replicas : Replicate(s, d)
        \/ Commit

Spec == Init /\ [][Next]_vars /\ WF_vars(Next)

\* Strong invariant: every committed prefix lives on a majority.
TypeOK == /\ logLen    \in [Replicas -> 0..MaxLog]
          /\ committed \in 0..MaxLog
          /\ Cardinality({r \in Replicas : logLen[r] >= committed}) >= Quorum
====
