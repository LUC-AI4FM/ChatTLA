---- MODULE FencingToken ----
(***************************************************************************)
(*  Fencing tokens (Martin Kleppmann).  A lock service hands out         *)
(*  monotone tokens; the storage layer rejects any write whose token is *)
(*  smaller than the largest token it has ever accepted.               *)
(*                                                                         *)
(*  Strong invariant: the storage layer's lastAccepted token is monotone*)
(*  and equals the maximum of the storage's accepted set.              *)
(***************************************************************************)
EXTENDS Naturals, FiniteSets

CONSTANTS Clients, MaxToken

VARIABLES nextToken, held, lastAccepted, accepted

vars == << nextToken, held, lastAccepted, accepted >>

NONE == 0

Init == /\ nextToken    = 1
        /\ held         = [c \in Clients |-> NONE]
        /\ lastAccepted = 0
        /\ accepted     = {}

\* Lock service issues the next monotone token to a client.
Issue(c) == /\ nextToken <= MaxToken
            /\ held'      = [held EXCEPT ![c] = nextToken]
            /\ nextToken' = nextToken + 1
            /\ UNCHANGED << lastAccepted, accepted >>

\* Storage accepts a write tagged with token tk only if tk >= lastAccepted.
Write(c) == /\ held[c] # NONE
            /\ held[c] >= lastAccepted
            /\ accepted'     = accepted \cup {held[c]}
            /\ lastAccepted' = held[c]
            /\ UNCHANGED << nextToken, held >>

\* Storage rejects a stale write (no state change at the storage layer).
Reject(c) == /\ held[c] # NONE
             /\ held[c] < lastAccepted
             /\ UNCHANGED vars

Next == \/ \E c \in Clients : Issue(c)
        \/ \E c \in Clients : Write(c)
        \/ \E c \in Clients : Reject(c)

Spec == Init /\ [][Next]_vars /\ WF_vars(Next)

\* Strong invariant: lastAccepted is the maximum of accepted (or 0).
Max(S) == IF S = {} THEN 0 ELSE CHOOSE x \in S : \A y \in S : y <= x

TypeOK == /\ nextToken    \in 1..(MaxToken + 1)
          /\ held         \in [Clients -> 0..MaxToken]
          /\ lastAccepted \in 0..MaxToken
          /\ accepted     \subseteq 1..MaxToken
          /\ lastAccepted = Max(accepted)
          /\ \A t \in accepted : t <= lastAccepted
====
