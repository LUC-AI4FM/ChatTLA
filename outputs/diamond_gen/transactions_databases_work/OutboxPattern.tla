---- MODULE OutboxPattern ----
(***************************************************************************)
(*  Transactional outbox.  Application writes to a DB row and an outbox  *)
(*  row in a single local transaction.  A relay process scans the       *)
(*  outbox and publishes each row to a downstream message bus.          *)
(*                                                                         *)
(*  Strong invariant: the set of published messages is always a subset   *)
(*  of the set of committed outbox rows -- and the relay never publishes *)
(*  before the local transaction commits.                                *)
(***************************************************************************)
EXTENDS Naturals, FiniteSets

CONSTANTS Msgs

VARIABLES dbRows, outbox, published

vars == << dbRows, outbox, published >>

Init == /\ dbRows    = {}
        /\ outbox    = {}
        /\ published = {}

\* Single local transaction: write to dbRows AND outbox atomically.
WriteLocal(m) ==
    /\ m \notin dbRows
    /\ dbRows' = dbRows \cup {m}
    /\ outbox' = outbox \cup {m}
    /\ UNCHANGED published

\* Relay publishes a committed outbox row to the downstream bus.
Publish(m) ==
    /\ m \in outbox
    /\ m \notin published
    /\ published' = published \cup {m}
    /\ UNCHANGED << dbRows, outbox >>

Next == \/ \E m \in Msgs : WriteLocal(m)
        \/ \E m \in Msgs : Publish(m)

Spec == Init /\ [][Next]_vars /\ WF_vars(Next)

\* Strong invariant: published \subseteq outbox = dbRows.
TypeOK == /\ dbRows    \subseteq Msgs
          /\ outbox    \subseteq Msgs
          /\ published \subseteq Msgs
          /\ outbox    = dbRows
          /\ published \subseteq outbox
====
