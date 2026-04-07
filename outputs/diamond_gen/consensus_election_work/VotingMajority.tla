---- MODULE VotingMajority ----
(***************************************************************************)
(* Simple majority voting among N voters with two candidates.  Each voter *)
(* casts at most one ballot; a winner is declared once a candidate has    *)
(* strictly more than N/2 votes.                                          *)
(* Safety: at most one winner, and the winner has > N/2 votes.            *)
(***************************************************************************)
EXTENDS Naturals, FiniteSets

Voters     == {1, 2, 3}
Candidates == {"A", "B"}
Half       == 1                          \* N/2 floor for N = 3

VARIABLES ballot, winner

vars == << ballot, winner >>

Init == /\ ballot = [v \in Voters |-> "none"]
        /\ winner = "none"

\* A voter casts a ballot for some candidate.
Vote(v, c) ==
    /\ ballot[v] = "none"
    /\ winner = "none"
    /\ ballot' = [ballot EXCEPT ![v] = c]
    /\ UNCHANGED winner

Tally(c) == Cardinality({v \in Voters : ballot[v] = c})

\* Declare c the winner once it has strictly more than half of the votes.
Declare(c) ==
    /\ winner = "none"
    /\ Tally(c) > Half
    /\ winner' = c
    /\ UNCHANGED ballot

\* Restart for the next election.
Reset ==
    /\ winner # "none"
    /\ ballot' = [v \in Voters |-> "none"]
    /\ winner' = "none"

Next == \/ \E v \in Voters : \E c \in Candidates : Vote(v, c)
        \/ \E c \in Candidates : Declare(c)
        \/ Reset

Spec == Init /\ [][Next]_vars

TypeOK ==
    /\ ballot \in [Voters -> Candidates \cup {"none"}]
    /\ winner \in Candidates \cup {"none"}

\* Strong safety: any declared winner actually has a majority of the votes,
\* and at most one candidate is the declared winner.
SafetyInv == (winner # "none") => (Tally(winner) > Half)
====
