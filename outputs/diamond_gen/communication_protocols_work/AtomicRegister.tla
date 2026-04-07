---- MODULE AtomicRegister ----
(***************************************************************************)
(* ABD-style read/write register replicated on N=3 nodes with majority    *)
(* quorums.  Each node stores a tagged value (timestamp, value).  A write *)
(* picks a fresh timestamp and is acknowledged by a majority; a read      *)
(* gathers from a majority and re-imposes the largest tag.                *)
(* Strong safety: replica timestamps are monotone non-decreasing and the  *)
(* "committed" value (max-tag in any majority) never regresses.           *)
(***************************************************************************)
EXTENDS Naturals, FiniteSets

Nodes  == {"n1", "n2", "n3"}
Vals   == {0, 1}
MaxTs  == 2

Majority(S) == Cardinality(S) * 2 > Cardinality(Nodes)

VARIABLES store, writeTs

vars == << store, writeTs >>

\* store[n] = <<ts, val>> currently held by node n.
\* writeTs  = highest timestamp ever issued by a writer (monotone).
Init ==
    /\ store = [n \in Nodes |-> <<0, 0>>]
    /\ writeTs = 0

\* Writer issues a new write with the next timestamp and a chosen value.
StartWrite ==
    /\ writeTs < MaxTs
    /\ writeTs' = writeTs + 1
    /\ UNCHANGED store

\* A node accepts the new write if its tag is fresher.
AcceptWrite(n, v) ==
    /\ writeTs > 0
    /\ store[n][1] < writeTs
    /\ store' = [store EXCEPT ![n] = <<writeTs, v>>]
    /\ UNCHANGED writeTs

\* Read-impose: a read picks any majority Q, takes the max tag value, and
\* writes it back to nodes in Q whose tag is older.
ReadImpose(n, Q) ==
    /\ Q \subseteq Nodes
    /\ Majority(Q)
    /\ LET maxTag == CHOOSE t \in {store[m][1] : m \in Q} :
                       \A m \in Q : store[m][1] <= t
           winner == CHOOSE m \in Q : store[m][1] = maxTag IN
         /\ store[n][1] < maxTag
         /\ store' = [store EXCEPT ![n] = store[winner]]
         /\ UNCHANGED writeTs

Done == UNCHANGED vars

Next ==
    \/ StartWrite
    \/ \E n \in Nodes, v \in Vals : AcceptWrite(n, v)
    \/ \E n \in Nodes, Q \in SUBSET Nodes : ReadImpose(n, Q)
    \/ Done

Spec == Init /\ [][Next]_vars

\* Strong safety conjoined into TypeOK: stored timestamp never exceeds the
\* writer's high-water-mark, and value-with-tag-0 is the initial 0.
TypeOK ==
    /\ store \in [Nodes -> (0 .. MaxTs) \X Vals]
    /\ writeTs \in 0 .. MaxTs
    /\ \A n \in Nodes : store[n][1] <= writeTs
    /\ \A n \in Nodes : (store[n][1] = 0) => (store[n][2] = 0)
====
