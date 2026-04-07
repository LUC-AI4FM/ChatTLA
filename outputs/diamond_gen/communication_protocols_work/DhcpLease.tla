---- MODULE DhcpLease ----
(***************************************************************************)
(* DHCP DORA: Discover, Offer, Request, Ack.                               *)
(* A bounded address pool, two clients.  Server tracks each address as     *)
(* free / offered / leased and to whom.                                    *)
(* Strong safety: no two clients hold the same address simultaneously.     *)
(***************************************************************************)
EXTENDS Naturals

Clients == {"c1", "c2"}
Addrs   == {"a1", "a2"}

VARIABLES cstate, owner, status, msgs

vars == << cstate, owner, status, msgs >>

ClientStates == {"init", "selecting", "requesting", "bound"}
AddrStatus   == {"free", "offered", "leased"}

NoOwner == "none"

Init ==
    /\ cstate = [c \in Clients |-> "init"]
    /\ owner  = [a \in Addrs   |-> NoOwner]
    /\ status = [a \in Addrs   |-> "free"]
    /\ msgs   = {}

\* 1. Client broadcasts DISCOVER.
Discover(c) ==
    /\ cstate[c] = "init"
    /\ cstate' = [cstate EXCEPT ![c] = "selecting"]
    /\ msgs' = msgs \cup {<<"discover", c>>}
    /\ UNCHANGED << owner, status >>

\* 2. Server picks a free address and offers it to c.
Offer(c, a) ==
    /\ <<"discover", c>> \in msgs
    /\ status[a] = "free"
    /\ status' = [status EXCEPT ![a] = "offered"]
    /\ owner'  = [owner  EXCEPT ![a] = c]
    /\ msgs' = (msgs \ {<<"discover", c>>}) \cup {<<"offer", c, a>>}
    /\ UNCHANGED cstate

\* 3. Client requests the offered address.
Request(c, a) ==
    /\ cstate[c] = "selecting"
    /\ <<"offer", c, a>> \in msgs
    /\ cstate' = [cstate EXCEPT ![c] = "requesting"]
    /\ msgs' = (msgs \ {<<"offer", c, a>>}) \cup {<<"request", c, a>>}
    /\ UNCHANGED << owner, status >>

\* 4. Server acks: address transitions to leased.
Ack(c, a) ==
    /\ <<"request", c, a>> \in msgs
    /\ status[a] = "offered"
    /\ owner[a] = c
    /\ status' = [status EXCEPT ![a] = "leased"]
    /\ cstate' = [cstate EXCEPT ![c] = "bound"]
    /\ msgs' = msgs \ {<<"request", c, a>>}
    /\ UNCHANGED owner

Done == UNCHANGED vars

Next ==
    \/ \E c \in Clients : Discover(c)
    \/ \E c \in Clients, a \in Addrs : Offer(c, a)
    \/ \E c \in Clients, a \in Addrs : Request(c, a)
    \/ \E c \in Clients, a \in Addrs : Ack(c, a)
    \/ Done

Spec == Init /\ [][Next]_vars

\* Strong safety conjoined into TypeOK: every leased / offered address
\* has a unique owner (no double allocation).
TypeOK ==
    /\ cstate \in [Clients -> ClientStates]
    /\ owner  \in [Addrs   -> Clients \cup {NoOwner}]
    /\ status \in [Addrs   -> AddrStatus]
    /\ \A a \in Addrs : (status[a] = "free") <=> (owner[a] = NoOwner)
    /\ \A a, b \in Addrs :
         (a # b /\ status[a] # "free" /\ status[b] # "free")
            => (owner[a] # owner[b])
    /\ \A c \in Clients :
         (cstate[c] = "bound") =>
            (\E a \in Addrs : owner[a] = c /\ status[a] = "leased")
====
