---- MODULE IdempotentWrite ----
(***************************************************************************)
(*  Idempotent write protocol.  Each request carries a unique id; the     *)
(*  server records ids it has already processed and silently ignores      *)
(*  retries.  Even with retries, the effect of a request is applied at    *)
(*  most once.                                                             *)
(*                                                                         *)
(*  Strong invariant: the multiset of effects equals the set of distinct  *)
(*  request ids the server has accepted -- duplicates do not contribute. *)
(***************************************************************************)
EXTENDS Naturals, FiniteSets

CONSTANTS Reqs

VARIABLES inFlight, processed, counter

vars == << inFlight, processed, counter >>

Init == /\ inFlight  = {}
        /\ processed = {}
        /\ counter   = 0

\* Client emits a request id (possibly more than once).
Send(r) == /\ inFlight' = inFlight \cup {r}
           /\ UNCHANGED << processed, counter >>

\* The server processes a request: if its id has not been seen, apply
\* the effect (increment counter) and record the id.  Otherwise, dedupe.
Process(r) == /\ r \in inFlight
              /\ \/ /\ r \notin processed
                    /\ processed' = processed \cup {r}
                    /\ counter'   = counter + 1
                 \/ /\ r \in processed
                    /\ UNCHANGED << processed, counter >>
              /\ UNCHANGED inFlight

Next == \/ \E r \in Reqs : Send(r)
        \/ \E r \in Reqs : Process(r)

Spec == Init /\ [][Next]_vars /\ WF_vars(Next)

\* Strong invariant: counter equals the number of distinct processed ids.
\* Conjoined into TypeOK so the mutation test catches the lost equality.
TypeOK == /\ inFlight  \subseteq Reqs
          /\ processed \subseteq Reqs
          /\ counter   \in 0..Cardinality(Reqs)
          /\ counter   = Cardinality(processed)
          /\ processed \subseteq inFlight
====
