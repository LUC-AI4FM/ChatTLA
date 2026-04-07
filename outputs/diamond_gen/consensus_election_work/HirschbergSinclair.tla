---- MODULE HirschbergSinclair ----
(***************************************************************************)
(* Hirschberg-Sinclair O(N log N) bidirectional ring leader election.     *)
(* Each process repeatedly probes outward in both directions, doubling    *)
(* the probe distance every phase.  A probe survives a phase only if it   *)
(* reaches a process whose id is smaller; otherwise the probe dies.  The  *)
(* process whose probe survives all log N phases is the unique leader.    *)
(*                                                                         *)
(* We model this abstractly: per process, the highest phase its id has    *)
(* reached.  The maximum id always advances; smaller ids fall behind.     *)
(* Safety: at most one leader; the leader has the maximum id.             *)
(***************************************************************************)
EXTENDS Naturals, FiniteSets

N == 3
Procs == 1..N
MaxPhase == 2          \* ceil(log2 N) + 1, kept tiny on purpose

VARIABLES phase, leader

vars == << phase, leader >>

Init == /\ phase  = [i \in Procs |-> 0]
        /\ leader = 0

\* A process can only continue advancing if it still wins the global
\* tournament: its id is the maximum among all processes that have not yet
\* fallen behind (i.e., its id is at least as large as every other process
\* whose phase has reached at least the same level).
Advance(i) ==
    /\ leader = 0
    /\ phase[i] < MaxPhase
    /\ \A j \in Procs : j > i => phase[j] < phase[i]
    /\ phase' = [phase EXCEPT ![i] = @ + 1]
    /\ UNCHANGED leader

\* The unique process that reaches the maximum phase declares itself leader.
Elect(i) ==
    /\ leader = 0
    /\ phase[i] = MaxPhase
    /\ \A j \in Procs : j # i => phase[j] < MaxPhase
    /\ leader' = i
    /\ UNCHANGED phase

\* Restart the protocol after a leader is elected to keep the state space
\* finite without leaving terminal states.
Reset ==
    /\ leader # 0
    /\ phase'  = [i \in Procs |-> 0]
    /\ leader' = 0

Next == \/ \E i \in Procs : Advance(i)
        \/ \E i \in Procs : Elect(i)
        \/ Reset

Spec == Init /\ [][Next]_vars

TypeOK ==
    /\ phase  \in [Procs -> 0..MaxPhase]
    /\ leader \in 0..N

\* Strong safety: only the maximum id can ever become leader.
SafetyInv == (leader # 0) => (\A j \in Procs : j <= leader)
====
