---- MODULE Numa ----
(***************************************************************************)
(* Two-socket NUMA model.                                                 *)
(*                                                                         *)
(* Each socket has its own local memory cell.  A write to a remote       *)
(* socket goes through a propagation phase: the value is staged in a    *)
(* "pending" slot until propagation completes (modeling cross-socket    *)
(* latency).  A remote read sees the latest globally-propagated value;  *)
(* once propagation has completed both sockets agree.                    *)
(***************************************************************************)
EXTENDS Naturals, FiniteSets

CONSTANTS Sockets, MaxIssue

VARIABLES mem,         \* mem[s] : value visible at socket s
          pending,     \* pending[s] : value being propagated TO socket s, or 0
          issued       \* number of remote writes issued so far

vars == << mem, pending, issued >>

Vals == 1..2

Init == /\ mem     = [s \in Sockets |-> 0]
        /\ pending = [s \in Sockets |-> 0]
        /\ issued  = 0

\* A local write at socket s installs the value in s's local memory and
\* posts a propagation request to every OTHER socket.
LocalWrite(s, v) ==
    /\ issued < MaxIssue
    /\ \A r \in Sockets \ {s} : pending[r] = 0
    /\ mem'     = [mem EXCEPT ![s] = v]
    /\ pending' = [r \in Sockets |->
                       IF r = s THEN 0 ELSE v]
    /\ issued'  = issued + 1

\* Propagation: a pending update at socket r is committed to r's local
\* memory.  Models cross-socket latency: the propagation may be delayed.
Propagate(r) ==
    /\ pending[r] # 0
    /\ mem'     = [mem     EXCEPT ![r] = pending[r]]
    /\ pending' = [pending EXCEPT ![r] = 0]
    /\ UNCHANGED issued

\* Idle.
Idle == /\ issued = MaxIssue
        /\ \A r \in Sockets : pending[r] = 0
        /\ UNCHANGED vars

Next == \/ \E s \in Sockets, v \in Vals : LocalWrite(s, v)
        \/ \E r \in Sockets : Propagate(r)
        \/ Idle

Spec == Init /\ [][Next]_vars

\* --- Strong safety properties (folded into TypeOK) ---

\* When all propagations have settled (pending all zero) every socket    *)
\* must agree on memory.  This is the eventual-consistency contract.
QuiescentAgreement ==
    (\A s \in Sockets : pending[s] = 0) =>
        (\A s, t \in Sockets : mem[s] = mem[t])

TypeOK == /\ mem     \in [Sockets -> 0..2]
          /\ pending \in [Sockets -> 0..2]
          /\ issued  \in 0..MaxIssue
          /\ QuiescentAgreement
====
