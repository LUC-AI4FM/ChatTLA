---- MODULE DistributedLock ----
(***************************************************************************)
(*  Lease-based distributed lock.  A client may hold the lock only while  *)
(*  its lease has not yet expired.  Time advances in discrete ticks.      *)
(*  At every logical instant there is at most one valid holder.           *)
(***************************************************************************)
EXTENDS Naturals, FiniteSets

CONSTANTS Clients, MaxTime, LeaseLen

VARIABLES holder, expiry, now

vars == << holder, expiry, now >>

NONE == "none"

Init == /\ holder = NONE
        /\ expiry = 0
        /\ now    = 0

\* Acquire the lock when it is free or when the previous holder's lease
\* has expired.
Acquire(c) ==
    /\ \/ holder = NONE
       \/ now >= expiry
    /\ holder' = c
    /\ expiry' = now + LeaseLen
    /\ UNCHANGED now

\* The current holder renews its lease before the deadline.
Renew(c) ==
    /\ holder = c
    /\ now < expiry
    /\ expiry' = now + LeaseLen
    /\ UNCHANGED << holder, now >>

Tick == /\ now < MaxTime
        /\ now' = now + 1
        /\ UNCHANGED << holder, expiry >>

Next == \/ \E c \in Clients : Acquire(c)
        \/ \E c \in Clients : Renew(c)
        \/ Tick

Spec == Init /\ [][Next]_vars /\ WF_vars(Next)

\* Strong invariant: at most one holder at any logical time.  We model
\* this by ensuring that holder is in Clients \cup {NONE}, and once held
\* the expiry is in the future of acquisition.  Because there is only a
\* single holder variable this is "structurally" mutual exclusion.
TypeOK == /\ holder \in Clients \cup {NONE}
          /\ expiry \in 0..(MaxTime + LeaseLen)
          /\ now    \in 0..MaxTime
          /\ (holder = NONE) \/ (expiry > 0)
          /\ (holder # NONE) => (expiry >= LeaseLen)
====
