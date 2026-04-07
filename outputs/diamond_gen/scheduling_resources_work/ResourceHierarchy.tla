---- MODULE ResourceHierarchy ----
(***************************************************************************)
(* Resource hierarchy / lock ordering: N processes acquire R resources    *)
(* in a fixed total order (0 < 1 < ... < R-1).  A process must hold all  *)
(* lower-numbered resources it wants before requesting a higher one.     *)
(* Releasing is in reverse order.  This eliminates circular wait, so the *)
(* system is deadlock-free, and per-resource mutual exclusion holds.     *)
(*                                                                         *)
(* Safety: each resource is held by at most one process; if process p    *)
(* holds resource r > 0, it also holds every lower-numbered resource     *)
(* (the strict-order discipline).                                        *)
(***************************************************************************)
EXTENDS Naturals, FiniteSets

CONSTANT N

ASSUME N \in 1..3

Procs == 0..(N-1)
R == 2  \* number of resources
Resources == 0..(R-1)

VARIABLES holds  \* holds[r] = process holding resource r, or NoOne

NoOne == N
vars == << holds >>

Init == holds = [r \in Resources |-> NoOne]

\* Process p acquires resource r — must already hold all r' < r, and r free.
Acquire(p, r) == /\ holds[r] = NoOne
                 /\ \A r2 \in 0..(r-1) : holds[r2] = p
                 /\ holds' = [holds EXCEPT ![r] = p]

\* Process p releases the highest-numbered resource it currently holds.
ReleaseTop(p) == /\ \E r \in Resources : holds[r] = p
                 /\ LET top == CHOOSE r \in Resources :
                                 /\ holds[r] = p
                                 /\ \A r2 \in (r+1)..(R-1) : holds[r2] # p
                    IN  holds' = [holds EXCEPT ![top] = NoOne]

Next == (\E p \in Procs, r \in Resources : Acquire(p, r))
        \/ (\E p \in Procs : ReleaseTop(p))

Spec == Init /\ [][Next]_vars

\* Strong safety: ordering discipline — if a process holds a higher resource
\* it must also hold every lower-numbered one.
OrderingInv ==
  \A p \in Procs, r \in Resources :
    (holds[r] = p) => (\A r2 \in 0..(r-1) : holds[r2] = p)

TypeOK == /\ \A r \in Resources : holds[r] \in Procs \cup {NoOne}
          /\ OrderingInv
====
