---- MODULE TicketLock ----
EXTENDS Naturals, FiniteSets

CONSTANTS Procs, MaxTicket

\* next_ticket : the next ticket number to be issued
\* now_serving : the ticket currently allowed to enter the CS
\* ticket      : per-process ticket; 0 means "no ticket"
VARIABLES next_ticket, now_serving, ticket

vars == << next_ticket, now_serving, ticket >>

Init == /\ next_ticket = 0
        /\ now_serving = 0
        /\ ticket = [p \in Procs |-> 0]

\* Take a ticket: process draws the next ticket and increments the counter.
TakeTicket(p) == /\ ticket[p] = 0
                 /\ next_ticket < MaxTicket
                 /\ ticket' = [ticket EXCEPT ![p] = next_ticket + 1]
                 /\ next_ticket' = next_ticket + 1
                 /\ UNCHANGED now_serving

\* Release: holder advances now_serving and clears its ticket.
Release(p) == /\ ticket[p] # 0
              /\ ticket[p] = now_serving + 1
              /\ now_serving' = now_serving + 1
              /\ ticket' = [ticket EXCEPT ![p] = 0]
              /\ UNCHANGED next_ticket

\* Reset both counters once the lock has been fully drained.
Reset == /\ next_ticket = MaxTicket
         /\ now_serving = MaxTicket
         /\ \A p \in Procs : ticket[p] = 0
         /\ next_ticket' = 0
         /\ now_serving' = 0
         /\ UNCHANGED ticket

Next == \/ \E p \in Procs : TakeTicket(p)
        \/ \E p \in Procs : Release(p)
        \/ Reset

Spec == Init /\ [][Next]_vars

\* The set of processes currently in the critical section.
Holders == { p \in Procs : ticket[p] = now_serving + 1 }

\* Safety: at most one holder; now_serving never exceeds next_ticket;
\* tickets are unique among processes that hold one.
TicketSafe == /\ Cardinality(Holders) <= 1
              /\ now_serving <= next_ticket
              /\ \A p, q \in Procs :
                    (p # q /\ ticket[p] # 0 /\ ticket[q] # 0)
                       => ticket[p] # ticket[q]

TypeOK == /\ next_ticket \in 0..MaxTicket
          /\ now_serving \in 0..MaxTicket
          /\ ticket \in [Procs -> 0..MaxTicket]
          /\ TicketSafe
====
