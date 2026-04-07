---- MODULE StoreBufferTSO ----
(***************************************************************************)
(* Total Store Order (TSO) memory model with one core's store buffer.     *)
(*                                                                         *)
(* The core issues stores into a per-core FIFO store buffer.  Stores are  *)
(* not visible to global memory until they drain (in program order).  A   *)
(* sequence number tags each store so we can prove that drained stores    *)
(* reach memory in program order.                                          *)
(***************************************************************************)
EXTENDS Naturals, Sequences

CONSTANTS K, MaxIssue

VARIABLES sb,           \* store buffer: sequence of seq-numbers
          mem,          \* sequence of seq-numbers committed to memory in order
          nextSeq       \* next sequence number to assign

vars == << sb, mem, nextSeq >>

Init == /\ sb       = << >>
        /\ mem      = << >>
        /\ nextSeq  = 1

\* Core issues a store: append a new monotonically increasing seq-num to sb.
Issue == /\ Len(sb) < K
         /\ nextSeq <= MaxIssue
         /\ sb'      = Append(sb, nextSeq)
         /\ nextSeq' = nextSeq + 1
         /\ UNCHANGED mem

\* Drain: pop the head of sb and append to memory log.
Drain == /\ Len(sb) > 0
         /\ mem' = Append(mem, Head(sb))
         /\ sb'  = Tail(sb)
         /\ UNCHANGED nextSeq

\* Idle when nothing left to do.
Idle == /\ nextSeq > MaxIssue
        /\ Len(sb) = 0
        /\ UNCHANGED vars

Next == Issue \/ Drain \/ Idle

Spec == Init /\ [][Next]_vars

\* --- Strong safety properties (folded into TypeOK) ---

\* Buffer length is bounded by K.
Bounded == Len(sb) \in 0..K

\* Memory log is strictly increasing -- drained stores reach memory in
\* program order.  This is THE key TSO property for a single core.
MemInOrder ==
    \A i \in 1..(Len(mem) - 1) : mem[i] < mem[i+1]

\* Every entry in the store buffer is also strictly increasing and is
\* greater than every memory entry (the buffer holds in-flight stores).
SbInOrder ==
    /\ \A i \in 1..(Len(sb) - 1) : sb[i] < sb[i+1]
    /\ (Len(sb) > 0 /\ Len(mem) > 0) => mem[Len(mem)] < sb[1]

TypeOK == /\ sb      \in Seq(1..MaxIssue)
          /\ mem     \in Seq(1..MaxIssue)
          /\ nextSeq \in 1..(MaxIssue + 1)
          /\ Bounded
          /\ MemInOrder
          /\ SbInOrder
====
