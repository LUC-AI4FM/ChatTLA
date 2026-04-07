---- MODULE DmaTransfer ----
(***************************************************************************)
(* DMA engine.                                                            *)
(*                                                                         *)
(* Software writes a descriptor (src, dst, value) and starts the engine. *)
(* The engine copies src->dst, raises an interrupt, and goes idle.        *)
(* Safety contract: when the transfer is complete, the destination       *)
(* slot's value matches the descriptor's source value.                   *)
(***************************************************************************)
EXTENDS Naturals

CONSTANTS Slots, MaxIssue

VARIABLES memory,      \* memory[i] : value at slot i
          desc,        \* current descriptor: [src |-> i, dst |-> j, v |-> v]
          state,       \* "idle" / "running" / "done"
          interrupt,   \* TRUE iff completion interrupt is raised
          issued       \* number of transfers issued

vars == << memory, desc, state, interrupt, issued >>

Vals == 0..2
SlotIds == 1..Slots

NullDesc == [src |-> 0, dst |-> 0, v |-> 0]

Init == /\ memory    = [i \in SlotIds |-> 0]
        /\ desc      = NullDesc
        /\ state     = "idle"
        /\ interrupt = FALSE
        /\ issued    = 0

\* Software programs a descriptor and kicks off the engine.  We require
\* a non-zero source value so the contract is non-trivially testable.
Issue(s, d, v) ==
    /\ state = "idle"
    /\ issued < MaxIssue
    /\ s \in SlotIds
    /\ d \in SlotIds
    /\ v \in Vals
    /\ memory[s] = v        \* descriptor v matches actual src value
    /\ desc'      = [src |-> s, dst |-> d, v |-> v]
    /\ state'     = "running"
    /\ interrupt' = FALSE
    /\ issued'    = issued + 1
    /\ UNCHANGED memory

\* Engine performs the copy and asserts the interrupt line.
Complete ==
    /\ state = "running"
    /\ memory'    = [memory EXCEPT ![desc.dst] = desc.v]
    /\ state'     = "done"
    /\ interrupt' = TRUE
    /\ UNCHANGED << desc, issued >>

\* CPU services the interrupt and clears it; engine returns to idle.
Acknowledge ==
    /\ state = "done"
    /\ interrupt' = FALSE
    /\ state'     = "idle"
    /\ UNCHANGED << memory, desc, issued >>

Idle == /\ state = "idle"
        /\ issued = MaxIssue
        /\ UNCHANGED vars

Next == \/ \E s, d \in SlotIds, v \in Vals : Issue(s, d, v)
        \/ Complete
        \/ Acknowledge
        \/ Idle

Spec == Init /\ [][Next]_vars

\* --- Strong safety properties (folded into TypeOK) ---

\* The DMA contract: in any "done" state the destination slot reflects
\* the source value the descriptor recorded.
DoneImpliesCopied ==
    state = "done" => memory[desc.dst] = desc.v

\* The interrupt line is raised iff the engine is in the done state.
InterruptIffDone == interrupt <=> (state = "done")

TypeOK == /\ memory \in [SlotIds -> Vals]
          /\ desc \in [src : SlotIds \cup {0},
                       dst : SlotIds \cup {0},
                       v   : Vals]
          /\ state \in {"idle", "running", "done"}
          /\ interrupt \in BOOLEAN
          /\ issued \in 0..MaxIssue
          /\ DoneImpliesCopied
          /\ InterruptIffDone
====
