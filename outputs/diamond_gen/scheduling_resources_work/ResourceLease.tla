---- MODULE ResourceLease ----
(***************************************************************************)
(* Resource lease with bounded duration.  At most one process holds the  *)
(* lease.  The holder accumulates ticks; when ticks reach Lease the      *)
(* lease automatically expires and the resource becomes free.            *)
(*                                                                         *)
(* Safety: at most one holder; the holder's tick count never exceeds the *)
(* configured Lease bound.                                               *)
(***************************************************************************)
EXTENDS Naturals

CONSTANT N

ASSUME N \in 1..3

Procs == 0..(N-1)
Lease == 2  \* maximum lease duration in ticks

NoOne == N

VARIABLES holder, ticks

vars == << holder, ticks >>

Init == /\ holder = NoOne
        /\ ticks  = 0

Acquire(p) == /\ holder = NoOne
              /\ holder' = p
              /\ ticks'  = 0

Tick == /\ holder # NoOne
        /\ ticks < Lease
        /\ ticks' = ticks + 1
        /\ UNCHANGED holder

\* Lease expiry — automatically frees the resource at the bound.
Expire == /\ holder # NoOne
          /\ ticks = Lease
          /\ holder' = NoOne
          /\ ticks'  = 0

\* Voluntary release before expiry.
Release == /\ holder # NoOne
           /\ ticks < Lease
           /\ holder' = NoOne
           /\ ticks'  = 0

Next == (\E p \in Procs : Acquire(p)) \/ Tick \/ Expire \/ Release

Spec == Init /\ [][Next]_vars

LeaseInv == /\ ticks \in 0..Lease
            /\ ((holder = NoOne) => (ticks = 0))

TypeOK == /\ holder \in Procs \cup {NoOne} /\ LeaseInv
====
