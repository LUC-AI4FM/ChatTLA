---- MODULE FetchAndAddMutex ----
(***************************************************************************)
(* Ticket lock using fetch-and-add.  Each thread atomically draws a unique *)
(* ticket; the holder is the one whose ticket equals `serving`.  This is   *)
(* the standard FIFO mutex used in Linux qspinlock fast-path emulators.    *)
(***************************************************************************)
EXTENDS Naturals

N == 2
Procs == 1..N
MaxTicket == 4

VARIABLES pc, ticket, next_ticket, serving

vars == << pc, ticket, next_ticket, serving >>

Init == /\ pc          = [i \in Procs |-> "ncs"]
        /\ ticket      = [i \in Procs |-> 0]
        /\ next_ticket = 0
        /\ serving     = 0

\* Atomic fetch-and-add on next_ticket; remember our ticket.
Acquire(i) ==
    /\ pc[i] = "ncs"
    /\ next_ticket < MaxTicket
    /\ ticket'      = [ticket EXCEPT ![i] = next_ticket]
    /\ next_ticket' = next_ticket + 1
    /\ pc'          = [pc EXCEPT ![i] = "wait"]
    /\ UNCHANGED serving

\* Spin until our ticket is being served.
EnterCS(i) ==
    /\ pc[i] = "wait"
    /\ ticket[i] = serving
    /\ pc' = [pc EXCEPT ![i] = "cs"]
    /\ UNCHANGED << ticket, next_ticket, serving >>

Release(i) ==
    /\ pc[i] = "cs"
    /\ serving' = serving + 1
    /\ pc'      = [pc EXCEPT ![i] = "ncs"]
    /\ UNCHANGED << ticket, next_ticket >>

Idle == UNCHANGED vars

Next == \/ \E i \in Procs : Acquire(i) \/ EnterCS(i) \/ Release(i)
        \/ Idle

Spec == Init /\ [][Next]_vars

TypeOK ==
    /\ pc          \in [Procs -> {"ncs","wait","cs"}]
    /\ ticket      \in [Procs -> 0..MaxTicket]
    /\ next_ticket \in 0..MaxTicket
    /\ serving     \in 0..MaxTicket
    /\ \A i, j \in Procs : (i # j /\ pc[i] = "cs") => pc[j] # "cs"
====
