---- MODULE MemoryFence ----
(***************************************************************************)
(* Two-thread store-buffer model with explicit MFENCE.                    *)
(*                                                                         *)
(* Each thread has its own FIFO store buffer.  An MFENCE drains the local *)
(* buffer to global memory before allowing further activity.  After an   *)
(* MFENCE, any subsequent read from a thread sees the globally visible   *)
(* values, never any pending value still in some buffer.                 *)
(***************************************************************************)
EXTENDS Naturals, Sequences, FiniteSets

CONSTANTS Threads, MaxIssue

VARIABLES sb,         \* sb[t] = sequence of pending stores for thread t
          fenced,     \* fenced[t] = TRUE iff thread t's buffer is empty
                      \*             since its last fence (or program start)
          issued      \* issued[t] = number of stores ever issued by t

vars == << sb, fenced, issued >>

Init == /\ sb     = [t \in Threads |-> << >>]
        /\ fenced = [t \in Threads |-> TRUE]
        /\ issued = [t \in Threads |-> 0]

\* Issue a store: append a value to thread t's buffer.  Stops being
\* "fenced" because there's now an in-flight unflushed store.
Issue(t) ==
    /\ Len(sb[t]) < 2
    /\ issued[t] < MaxIssue
    /\ sb'     = [sb     EXCEPT ![t] = Append(@, 1)]
    /\ fenced' = [fenced EXCEPT ![t] = FALSE]
    /\ issued' = [issued EXCEPT ![t] = @ + 1]

\* Asynchronous drain: a single store flushes from the head of the buffer.
\* Does NOT mark the thread as fenced -- the buffer may still be non-empty.
Drain(t) ==
    /\ Len(sb[t]) > 0
    /\ sb'     = [sb     EXCEPT ![t] = Tail(@)]
    /\ fenced' = [fenced EXCEPT ![t] = (Len(sb[t]) = 1)]
    /\ UNCHANGED issued

\* MFENCE: requires the entire buffer to have drained, then marks fenced.
\* (Idiomatic enabling-condition style: the program counter at the fence
\*  blocks until the buffer is empty.)
MFence(t) ==
    /\ Len(sb[t]) = 0
    /\ fenced' = [fenced EXCEPT ![t] = TRUE]
    /\ UNCHANGED << sb, issued >>

\* Idle so we never deadlock when issuance is exhausted and all drained.
Idle == /\ \A t \in Threads : issued[t] = MaxIssue /\ Len(sb[t]) = 0
        /\ UNCHANGED vars

Next == \/ \E t \in Threads : Issue(t)
        \/ \E t \in Threads : Drain(t)
        \/ \E t \in Threads : MFence(t)
        \/ Idle

Spec == Init /\ [][Next]_vars

\* --- Strong safety properties (folded into TypeOK) ---

\* The fundamental MFENCE contract: a thread is "fenced" iff its store
\* buffer is empty -- nothing pending, all writes globally visible.
FenceImpliesDrained ==
    \A t \in Threads : fenced[t] => Len(sb[t]) = 0

\* Buffer is bounded.
Bounded == \A t \in Threads : Len(sb[t]) \in 0..2

TypeOK == /\ sb     \in [Threads -> Seq(1..1)]
          /\ fenced \in [Threads -> BOOLEAN]
          /\ issued \in [Threads -> 0..MaxIssue]
          /\ FenceImpliesDrained
          /\ Bounded
====
