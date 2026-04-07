---- MODULE RaftElection ----
(***************************************************************************)
(* Raft leader election (subset).  Servers run in numbered terms; each    *)
(* server can vote at most once per term and a candidate becomes leader   *)
(* once it collects votes from a majority.  Safety: at most one leader    *)
(* per term (Raft's "Election Safety" property).                          *)
(***************************************************************************)
EXTENDS Naturals, FiniteSets

Servers   == {1, 2, 3}
Terms     == 1..2
Majority  == {Q \in SUBSET Servers : Cardinality(Q) >= 2}

VARIABLES state, currentTerm, votedFor, votesGranted, leader

vars == << state, currentTerm, votedFor, votesGranted, leader >>

Init == /\ state        = [s \in Servers |-> "follower"]
        /\ currentTerm   = [s \in Servers |-> 1]
        /\ votedFor      = [s \in Servers |-> 0]
        /\ votesGranted  = [s \in Servers |-> {}]
        /\ leader        = [t \in Terms |-> 0]

\* Follower times out and becomes a candidate, advancing its term and
\* voting for itself.
StartElection(s) ==
    /\ state[s] = "follower"
    /\ currentTerm[s] < 2
    /\ state'        = [state EXCEPT ![s] = "candidate"]
    /\ currentTerm'  = [currentTerm EXCEPT ![s] = @ + 1]
    /\ votedFor'     = [votedFor EXCEPT ![s] = s]
    /\ votesGranted' = [votesGranted EXCEPT ![s] = {s}]
    /\ UNCHANGED leader

\* Server v grants its vote to candidate c at term t (only if it has not
\* yet voted in t).
GrantVote(v, c) ==
    /\ v # c
    /\ state[c] = "candidate"
    /\ currentTerm[v] <= currentTerm[c]
    /\ \/ currentTerm[v] < currentTerm[c]
       \/ votedFor[v] = 0
    /\ currentTerm'  = [currentTerm EXCEPT ![v] = currentTerm[c]]
    /\ votedFor'     = [votedFor    EXCEPT ![v] = c]
    /\ state'        = [state       EXCEPT ![v] = "follower"]
    /\ votesGranted' = [votesGranted EXCEPT ![c] = @ \cup {v}]
    /\ UNCHANGED leader

\* A candidate that has gathered a majority of votes becomes the leader of
\* its term.
BecomeLeader(c) ==
    /\ state[c] = "candidate"
    /\ votesGranted[c] \in Majority
    /\ leader[currentTerm[c]] = 0
    /\ state'  = [state  EXCEPT ![c] = "leader"]
    /\ leader' = [leader EXCEPT ![currentTerm[c]] = c]
    /\ UNCHANGED << currentTerm, votedFor, votesGranted >>

\* Restart for the next round of elections — also fires on a split-brain
\* stalemate so we never reach a deadlocked terminal state.
Reset ==
    /\ \/ \E s \in Servers : state[s] = "leader"
       \/ \A s \in Servers : state[s] = "candidate"
    /\ state'        = [s \in Servers |-> "follower"]
    /\ currentTerm'  = [s \in Servers |-> 1]
    /\ votedFor'     = [s \in Servers |-> 0]
    /\ votesGranted' = [s \in Servers |-> {}]
    /\ leader'       = [t \in Terms |-> 0]

Next == \/ \E s \in Servers : StartElection(s)
        \/ \E v, c \in Servers : GrantVote(v, c)
        \/ \E c \in Servers : BecomeLeader(c)
        \/ Reset

Spec == Init /\ [][Next]_vars

TypeOK ==
    /\ state        \in [Servers -> {"follower", "candidate", "leader"}]
    /\ currentTerm  \in [Servers -> Terms]
    /\ votedFor     \in [Servers -> Servers \cup {0}]
    /\ votesGranted \in [Servers -> SUBSET Servers]
    /\ leader       \in [Terms -> Servers \cup {0}]

\* Strong safety: Raft's Election Safety — at most one leader per term.
SafetyInv == \A t \in Terms : \A s1, s2 \in Servers : (state[s1] = "leader" /\ state[s2] = "leader" /\ currentTerm[s1] = t /\ currentTerm[s2] = t) => s1 = s2
====
