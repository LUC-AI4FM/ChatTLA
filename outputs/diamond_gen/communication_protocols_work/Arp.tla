---- MODULE Arp ----
(***************************************************************************)
(* Address Resolution Protocol with a bounded cache.                       *)
(* On a cache miss the host broadcasts an ARP request; the owner of the    *)
(* requested IP replies; the requester installs a binding in its cache.    *)
(* Strong safety: every cache entry agrees with the canonical IP->MAC      *)
(* binding (i.e. no spoofing, no inconsistent installs).                   *)
(***************************************************************************)
EXTENDS Naturals, FiniteSets

IPs  == {"ip1", "ip2", "ip3"}
MACs == {"m1",  "m2",  "m3"}

\* Canonical, ground-truth IP->MAC mapping the network agrees on.
Truth == [ip1 |-> "m1", ip2 |-> "m2", ip3 |-> "m3"]

VARIABLES cache, requests, replies

vars == << cache, requests, replies >>

NoMac == "none"

Init ==
    /\ cache = [ip \in IPs |-> NoMac]
    /\ requests = {}
    /\ replies = {}

\* Host needs ip and broadcasts a request.
SendRequest ==
    /\ \E ip \in IPs :
         /\ cache[ip] = NoMac
         /\ requests' = requests \cup {ip}
    /\ UNCHANGED << cache, replies >>

\* Owner of ip replies with the canonical mac.
SendReply ==
    /\ \E ip \in requests :
         /\ replies' = replies \cup {<<ip, Truth[ip]>>}
         /\ requests' = requests \ {ip}
    /\ UNCHANGED cache

\* Requester installs the reply.
InstallReply ==
    /\ \E r \in replies :
         /\ cache' = [cache EXCEPT ![r[1]] = r[2]]
         /\ replies' = replies \ {r}
    /\ UNCHANGED requests

\* Cache entries may be evicted (timeout).
EvictEntry ==
    /\ \E ip \in IPs :
         /\ cache[ip] # NoMac
         /\ cache' = [cache EXCEPT ![ip] = NoMac]
    /\ UNCHANGED << requests, replies >>

Done == UNCHANGED vars

Next == SendRequest \/ SendReply \/ InstallReply \/ EvictEntry \/ Done

Spec == Init /\ [][Next]_vars

\* Strong safety conjoined into TypeOK: every populated cache entry must
\* match the canonical binding; replies in flight must too.  This rules
\* out cache poisoning.
TypeOK ==
    /\ cache \in [IPs -> MACs \cup {NoMac}]
    /\ requests \subseteq IPs
    /\ replies \subseteq (IPs \X MACs)
    /\ \A ip \in IPs : (cache[ip] # NoMac) => (cache[ip] = Truth[ip])
    /\ \A r \in replies : r[2] = Truth[r[1]]
====
