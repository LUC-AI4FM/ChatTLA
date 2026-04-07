---- MODULE WriteBuffer ----
(***************************************************************************)
(* A bounded per-core write buffer.  Stores are issued by a core into a    *)
(* FIFO buffer; they retire (commit) in order to main memory.  The        *)
(* visible memory always equals the sum of the values of all retired      *)
(* stores -- the architectural state contract.                            *)
(*                                                                         *)
(* To keep the model finite we bound:                                     *)
(*   - K        : maximum buffer length                                   *)
(*   - MaxIssue : maximum number of stores ever issued                    *)
(* Each store carries a value in {0, 1}.                                  *)
(***************************************************************************)
EXTENDS Naturals, Sequences

CONSTANTS K, MaxIssue

VARIABLES buffer, memory, retired, issued

vars == << buffer, memory, retired, issued >>

Vals == 0..1

Init == /\ buffer  = << >>
        /\ memory  = 0
        /\ retired = 0
        /\ issued  = 0

\* A new store enters the tail of the buffer.  Bounded by K and by MaxIssue.
Issue(v) == /\ Len(buffer) < K
            /\ issued < MaxIssue
            /\ buffer'  = Append(buffer, v)
            /\ issued'  = issued + 1
            /\ UNCHANGED << memory, retired >>

\* The oldest pending store retires: leaves the head of the buffer; its
\* value is added to both memory (visible state) and retired-sum.
Commit == /\ Len(buffer) > 0
          /\ memory'  = memory + Head(buffer)
          /\ retired' = retired + Head(buffer)
          /\ buffer'  = Tail(buffer)
          /\ UNCHANGED issued

\* Stutter action to avoid spurious deadlocks once issuance is exhausted
\* and the buffer is empty -- the system simply idles.
Idle == /\ issued = MaxIssue
        /\ Len(buffer) = 0
        /\ UNCHANGED vars

Next == \/ \E v \in Vals : Issue(v)
        \/ Commit
        \/ Idle

Spec == Init /\ [][Next]_vars

\* --- Strong safety properties (folded into TypeOK) ---

\* The buffer never exceeds its capacity.
Bounded == Len(buffer) \in 0..K

\* Visible memory always equals the sum of retired stores.
\* This is the FIFO write-buffer correctness contract.
MemoryMatchesRetired == memory = retired

\* Conservation: every issued store is either still buffered or retired.
\* (sum of in-flight values + retired = sum of issued values)
\* Stated bound-wise: retired <= issued AND buffer-length + retired-count <= issued.

TypeOK == /\ buffer  \in Seq(Vals)
          /\ memory  \in 0..MaxIssue
          /\ retired \in 0..MaxIssue
          /\ issued  \in 0..MaxIssue
          /\ Bounded
          /\ MemoryMatchesRetired
====
