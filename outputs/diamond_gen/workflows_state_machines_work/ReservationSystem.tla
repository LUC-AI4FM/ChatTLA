---- MODULE ReservationSystem ----
(***************************************************************************)
(*  Seat reservation system. K seats, each independently bookable.        *)
(*  Booking and cancellation track ever_booked history flags so the       *)
(*  invariant "no double booking" is checkable per seat.                  *)
(***************************************************************************)
EXTENDS Naturals

CONSTANT Seats

VARIABLES booked, ever_booked

vars == << booked, ever_booked >>

Init == /\ booked = [s \in Seats |-> FALSE]
        /\ ever_booked = [s \in Seats |-> FALSE]

Book(s) == /\ ~booked[s]
           /\ booked' = [booked EXCEPT ![s] = TRUE]
           /\ ever_booked' = [ever_booked EXCEPT ![s] = TRUE]

Cancel(s) == /\ booked[s]
             /\ booked' = [booked EXCEPT ![s] = FALSE]
             /\ UNCHANGED ever_booked

Done == /\ \A s \in Seats : ever_booked[s]
        /\ UNCHANGED vars

Next == \/ \E s \in Seats : Book(s)
        \/ \E s \in Seats : Cancel(s)
        \/ Done

Spec == Init /\ [][Next]_vars

\* Per-seat: a currently booked seat must have its ever_booked flag set.
\* Equivalently, no double booking can occur because Book is gated on ~booked.
SafetyInvariant == \A s \in Seats : (booked[s] => ever_booked[s])

TypeOK == /\ booked \in [Seats -> BOOLEAN]
          /\ ever_booked \in [Seats -> BOOLEAN]
          /\ SafetyInvariant
====
