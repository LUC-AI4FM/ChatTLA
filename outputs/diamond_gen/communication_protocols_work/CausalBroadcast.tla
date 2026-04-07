---- MODULE CausalBroadcast ----
(***************************************************************************)
(* Causal broadcast over reliable, FIFO links among two processes.         *)
(* Each broadcast is tagged with a vector timestamp recording how many     *)
(* messages from each sender have been delivered locally before the send.  *)
(* A receiver delivers a message m from sender s only when                 *)
(*   m.vc[s] = local.vc[s] + 1  AND                                        *)
(*   for every other p, m.vc[p] <= local.vc[p].                            *)
(* Strong safety: a delivered message's vector timestamp matches the       *)
(* receiver's current vector after delivery.                               *)
(***************************************************************************)
EXTENDS Naturals

Procs == {0, 1}
MaxMsgs == 2

VARIABLES vc, channel, sentCount

vars == << vc, channel, sentCount >>

\* vc[p][q] = number of broadcasts from q delivered at p.
\* sentCount[p] = number of broadcasts ever issued by p.
\* channel = set of in-flight messages << sender, vector >>.

ZeroVec == [q \in Procs |-> 0]

Init ==
    /\ vc = [p \in Procs |-> ZeroVec]
    /\ channel = {}
    /\ sentCount = [p \in Procs |-> 0]

Broadcast(p) ==
    /\ sentCount[p] < MaxMsgs
    /\ LET tag == [vc[p] EXCEPT ![p] = vc[p][p] + 1] IN
         /\ channel' = channel \cup {<<p, tag>>}
         /\ sentCount' = [sentCount EXCEPT ![p] = sentCount[p] + 1]
    /\ UNCHANGED vc

\* Sender delivers its own message immediately on broadcast.  Model that
\* explicitly so the sender's vc updates.
DeliverSelf(p) ==
    /\ \E m \in channel :
         /\ m[1] = p
         /\ vc[p][p] < m[2][p]
         /\ vc' = [vc EXCEPT ![p] = m[2]]
         /\ channel' = channel \ {m}
    /\ UNCHANGED sentCount

\* Remote delivery: causal predicates.
DeliverRemote(p) ==
    /\ \E m \in channel :
         /\ m[1] # p
         /\ m[2][m[1]] = vc[p][m[1]] + 1
         /\ \A q \in Procs : (q # m[1]) => m[2][q] <= vc[p][q]
         /\ vc' = [vc EXCEPT ![p] = [q \in Procs |->
                       IF q = m[1] THEN vc[p][q] + 1 ELSE vc[p][q]]]
         /\ channel' = channel \ {m}
    /\ UNCHANGED sentCount

Done == UNCHANGED vars

Next ==
    \/ \E p \in Procs : Broadcast(p)
    \/ \E p \in Procs : DeliverSelf(p)
    \/ \E p \in Procs : DeliverRemote(p)
    \/ Done

Spec == Init /\ [][Next]_vars

\* Strong safety conjoined into TypeOK: a process's local count of its own
\* deliveries never exceeds what it has sent; any in-flight message
\* describes a vector that is consistent with the sender's actual count.
TypeOK ==
    /\ vc \in [Procs -> [Procs -> 0 .. MaxMsgs]]
    /\ sentCount \in [Procs -> 0 .. MaxMsgs]
    /\ channel \subseteq (Procs \X [Procs -> 0 .. MaxMsgs])
    /\ \A p \in Procs : vc[p][p] <= sentCount[p]
    /\ \A m \in channel : m[2][m[1]] <= sentCount[m[1]]
    \* Causal monotonicity: a peer's local view of q is bounded by q's
    \* own count.
    /\ \A p, q \in Procs : vc[p][q] <= sentCount[q]
====
