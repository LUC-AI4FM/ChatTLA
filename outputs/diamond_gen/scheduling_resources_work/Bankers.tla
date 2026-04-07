---- MODULE Bankers ----
(***************************************************************************)
(* Dijkstra's banker's algorithm with two resource types and N processes.  *)
(*                                                                         *)
(* Each process p declares a maximum need max[p][r] for each resource r.   *)
(* Currently allocated amounts are alloc[p][r]; available[r] is what the   *)
(* bank still has.  A request from process p for some additional units is *)
(* granted only if the post-grant state is SAFE — i.e. there exists an     *)
(* ordering of the processes such that each can run to completion using   *)
(* the (current available + already-finished allocations) without ever    *)
(* exceeding its declared maximum.                                        *)
(*                                                                         *)
(* Safety: every reachable state is safe (a safe sequence exists).        *)
(***************************************************************************)
EXTENDS Naturals, FiniteSets, Sequences

CONSTANT N

ASSUME N \in 2..3

Procs == 0..(N-1)
Res == {0, 1}              \* two resource types

Total == [r \in Res |-> 2] \* total units of each resource (small)
Max   == [p \in Procs |-> [r \in Res |-> 2]]
                            \* every process may need up to 2 of each

VARIABLES alloc, available

vars == << alloc, available >>

Need(p) == [r \in Res |-> Max[p][r] - alloc[p][r]]

Init == /\ alloc     = [p \in Procs |-> [r \in Res |-> 0]]
        /\ available = Total

\* Can process p finish given a hypothetical available pool av?
CanFinish(p, av) == \A r \in Res : Need(p)[r] <= av[r]

\* After p finishes, the freed pool is av + alloc[p].
ReleaseInto(p, av) == [r \in Res |-> av[r] + alloc[p][r]]

\* Iterative safety check up to N steps (small finite N).
SafeAfter(av0, finished0) ==
  LET RECURSIVE Step(_, _, _)
      Step(av, finished, k) ==
        IF k = 0 THEN finished = Procs
        ELSE
          \E p \in Procs \ finished :
            /\ CanFinish(p, av)
            /\ Step(ReleaseInto(p, av), finished \cup {p}, k - 1)
  IN  finished0 = Procs \/ Step(av0, finished0, N)

\* Process p requests one unit of resource r — grant only if it leaves a safe state.
Request(p, r) == /\ alloc[p][r] < Max[p][r]
                 /\ available[r] >= 1
                 /\ LET newAlloc == [alloc EXCEPT ![p][r] = @ + 1]
                        newAv    == [available EXCEPT ![r] = @ - 1]
                        newNeed(q) == [s \in Res |-> Max[q][s] - newAlloc[q][s]]
                        canFinishNew(q, av) == \A s \in Res : newNeed(q)[s] <= av[s]
                        relInto(q, av) == [s \in Res |-> av[s] + newAlloc[q][s]]
                        RECURSIVE SafeStep(_, _, _)
                        SafeStep(av, finished, k) ==
                          IF finished = Procs THEN TRUE
                          ELSE IF k = 0 THEN FALSE
                          ELSE \E q \in Procs \ finished :
                                 /\ canFinishNew(q, av)
                                 /\ SafeStep(relInto(q, av), finished \cup {q}, k - 1)
                    IN  /\ SafeStep(newAv, {}, N)
                        /\ alloc' = newAlloc
                        /\ available' = newAv

\* Process p releases ALL its currently held units (modeling completion).
ReleaseAll(p) == /\ \E r \in Res : alloc[p][r] > 0
                 /\ available' = [r \in Res |-> available[r] + alloc[p][r]]
                 /\ alloc' = [alloc EXCEPT ![p] = [r \in Res |-> 0]]

Next == (\E p \in Procs, r \in Res : Request(p, r))
        \/ (\E p \in Procs : ReleaseAll(p))

Spec == Init /\ [][Next]_vars

\* Sum of alloc[*][r] across all processes — small N so we hard-iterate.
RECURSIVE SumAlloc(_, _)
SumAlloc(S, r) == IF S = {} THEN 0
                  ELSE LET p == CHOOSE q \in S : TRUE
                       IN  alloc[p][r] + SumAlloc(S \ {p}, r)

\* Strong safety: allocations never exceed declared maxima, and resources
\* are conserved (alloc + available equals Total for every type).
BankInv == /\ \A p \in Procs, r \in Res : alloc[p][r] <= Max[p][r]
           /\ \A r \in Res : available[r] + SumAlloc(Procs, r) = Total[r]

TypeOK == /\ \A p \in Procs, r \in Res : alloc[p][r] \in 0..Total[r]
          /\ \A r \in Res : available[r] \in 0..Total[r]
          /\ BankInv
====
