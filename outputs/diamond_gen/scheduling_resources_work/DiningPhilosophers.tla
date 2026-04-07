---- MODULE DiningPhilosophers ----
(***************************************************************************)
(* Dijkstra's dining philosophers with N philosophers and N forks.         *)
(* All philosophers pick up the lower-numbered fork first EXCEPT           *)
(* philosopher 0, who picks up the higher-numbered fork first.  This       *)
(* asymmetry breaks the cyclic wait and prevents deadlock.                 *)
(*                                                                         *)
(* Safety: no two adjacent philosophers eat simultaneously; each fork is   *)
(* held by at most one philosopher.                                        *)
(***************************************************************************)
EXTENDS Naturals

CONSTANT N

ASSUME N \in 3..4

Phils == 0..(N-1)
Forks == 0..(N-1)

\* The two forks adjacent to philosopher i.
LeftFork(i)  == i
RightFork(i) == (i + 1) % N

\* Asymmetric ordering: philosopher 0 takes its higher-numbered fork first.
FirstFork(i)  == IF i = 0 THEN RightFork(i) ELSE LeftFork(i)
SecondFork(i) == IF i = 0 THEN LeftFork(i)  ELSE RightFork(i)

VARIABLES state, holds

\* state[i] in {"thinking","hungry","oneFork","eating"}
vars == << state, holds >>

NoOne == N

Init == /\ state = [i \in Phils |-> "thinking"]
        /\ holds = [f \in Forks |-> NoOne]

GetHungry(i) == /\ state[i] = "thinking"
                /\ state' = [state EXCEPT ![i] = "hungry"]
                /\ UNCHANGED holds

PickFirst(i) == /\ state[i] = "hungry"
                /\ holds[FirstFork(i)] = NoOne
                /\ holds' = [holds EXCEPT ![FirstFork(i)] = i]
                /\ state' = [state EXCEPT ![i] = "oneFork"]

PickSecond(i) == /\ state[i] = "oneFork"
                 /\ holds[SecondFork(i)] = NoOne
                 /\ holds' = [holds EXCEPT ![SecondFork(i)] = i]
                 /\ state' = [state EXCEPT ![i] = "eating"]

Release(i) == /\ state[i] = "eating"
              /\ holds' = [f \in Forks |->
                              IF holds[f] = i THEN NoOne ELSE holds[f]]
              /\ state' = [state EXCEPT ![i] = "thinking"]

Next == \E i \in Phils :
          GetHungry(i) \/ PickFirst(i) \/ PickSecond(i) \/ Release(i)

Spec == Init /\ [][Next]_vars

\* Strong safety: no two adjacent philosophers eat simultaneously.
NoAdjacentEatingInv ==
  \A i \in Phils :
    (state[i] = "eating") => (state[(i + 1) % N] # "eating")

TypeOK == /\ \A i \in Phils : state[i] \in {"thinking","hungry","oneFork","eating"}
          /\ \A f \in Forks : holds[f] \in Phils \cup {NoOne}
          /\ NoAdjacentEatingInv
====
