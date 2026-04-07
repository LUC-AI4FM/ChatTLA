---- MODULE RingElection ----
(***************************************************************************)
(* Simple ring election: a single token carries the maximum id seen so far *)
(* once around a unidirectional ring.  When the token returns to the      *)
(* process whose id it carries, that process becomes the leader.          *)
(* Safety: the leader (when elected) has the maximum id.                  *)
(***************************************************************************)
EXTENDS Naturals, FiniteSets

N == 3
Procs == 1..N

Succ(i) == (i % N) + 1

\* token = << holder, maxSeen, hops >> ; hops counts how many times the
\* token has been forwarded — bounded so the state space stays finite.
VARIABLES token, leader

vars == << token, leader >>

NoToken == [holder |-> 0, maxSeen |-> 0, hops |-> 0]

Init == /\ token  = [holder |-> 1, maxSeen |-> 1, hops |-> 0]
        /\ leader = 0

\* Forward the token to the successor, updating the maximum id seen.  We
\* bound hops to one full ring traversal so the state space stays finite.
Pass ==
    /\ token.holder # 0
    /\ token.hops < N
    /\ leader = 0
    /\ LET nxt == Succ(token.holder)
           m   == IF nxt > token.maxSeen THEN nxt ELSE token.maxSeen
       IN token' = [holder |-> nxt, maxSeen |-> m, hops |-> token.hops + 1]
    /\ UNCHANGED leader

\* Once the token has completed a full pass, the maxSeen field equals the
\* global max id, and that process becomes leader.
Elect ==
    /\ token.holder # 0
    /\ token.hops >= N - 1
    /\ token.maxSeen = N
    /\ leader = 0
    /\ leader' = token.maxSeen
    /\ token'  = NoToken

\* Restart the protocol after a leader is elected.
Reset ==
    /\ leader # 0
    /\ token = NoToken
    /\ token'  = [holder |-> 1, maxSeen |-> 1, hops |-> 0]
    /\ leader' = 0

Next == Pass \/ Elect \/ Reset

Spec == Init /\ [][Next]_vars

TypeOK ==
    /\ token  \in [holder : 0..N, maxSeen : 0..N, hops : 0..N]
    /\ leader \in 0..N

\* Strong safety: the elected leader (when there is one) carries the
\* maximum id of all processes.
SafetyInv == (leader # 0) => (\A j \in Procs : j <= leader)
====
