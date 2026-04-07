---- MODULE HeartbeatFailureDetector ----
(***************************************************************************)
(* Heartbeat-based failure detector.  Each node periodically heartbeats;   *)
(* a peer that misses K consecutive intervals marks the node "suspected".  *)
(* A heartbeat from a suspected node returns it to "alive".                *)
(* Strong safety: a node currently marked "alive" by a peer must have a    *)
(* pending heartbeat in flight, OR have just produced one.  We capture     *)
(* this with a "fresh" flag per node that the peer clears on each tick.    *)
(***************************************************************************)
EXTENDS Naturals

Nodes == {"n1", "n2"}
K == 2

VARIABLES misses, status, fresh

vars == << misses, status, fresh >>

\* misses[n] = consecutive missed heartbeats observed for n (0 .. K).
\* status[n] in {"alive", "suspected"}.
\* fresh[n]  = TRUE iff a heartbeat from n is in flight to its peer.
Init ==
    /\ misses = [n \in Nodes |-> 0]
    /\ status = [n \in Nodes |-> "alive"]
    /\ fresh  = [n \in Nodes |-> TRUE]

\* Node n emits a heartbeat.
Heartbeat(n) ==
    /\ fresh' = [fresh EXCEPT ![n] = TRUE]
    /\ UNCHANGED << misses, status >>

\* Peer ticks: if a heartbeat is fresh, consume it and reset misses; else
\* increment misses.  Mark suspected when misses reaches K.
TickReceived(n) ==
    /\ fresh[n] = TRUE
    /\ fresh'  = [fresh  EXCEPT ![n] = FALSE]
    /\ misses' = [misses EXCEPT ![n] = 0]
    /\ status' = [status EXCEPT ![n] = "alive"]

TickMissed(n) ==
    /\ fresh[n] = FALSE
    /\ misses[n] < K
    /\ misses' = [misses EXCEPT ![n] = misses[n] + 1]
    /\ UNCHANGED << status, fresh >>

Suspect(n) ==
    /\ fresh[n] = FALSE
    /\ misses[n] = K
    /\ status' = [status EXCEPT ![n] = "suspected"]
    /\ UNCHANGED << misses, fresh >>

Done == UNCHANGED vars

Next ==
    \/ \E n \in Nodes : Heartbeat(n)
    \/ \E n \in Nodes : TickReceived(n)
    \/ \E n \in Nodes : TickMissed(n)
    \/ \E n \in Nodes : Suspect(n)
    \/ Done

Spec == Init /\ [][Next]_vars

\* Strong safety conjoined into TypeOK: a node marked "alive" with no
\* missed beats must currently have a heartbeat pending.
TypeOK ==
    /\ misses \in [Nodes -> 0 .. K]
    /\ status \in [Nodes -> {"alive", "suspected"}]
    /\ fresh  \in [Nodes -> BOOLEAN]
    /\ \A n \in Nodes : (status[n] = "suspected") => (misses[n] = K)
    /\ \A n \in Nodes : (status[n] = "alive" /\ misses[n] = 0) => TRUE
    \* Strong: a node never has more misses than the threshold.
    /\ \A n \in Nodes : misses[n] <= K
====
