---- MODULE BlocksWorld ----
(***************************************************************************)
(* Classic STRIPS blocks world with 3 blocks {a, b, c}, a single robot   *)
(* hand, and a table.                                                    *)
(*                                                                         *)
(* For each block we record what is "on" it: another block, the table,   *)
(* or the hand (if it is being held).  The hand may hold at most one     *)
(* block.  Operators: pickup, putdown, stack, unstack.                   *)
(***************************************************************************)
EXTENDS Naturals, FiniteSets

VARIABLES on, hand   \* on : Block -> Support ; hand : Block | "empty"

vars == << on, hand >>

Blocks   == {"a", "b", "c"}
Supports == Blocks \cup {"table", "hand"}

\* A block is clear iff no other block sits on it.
Clear(b) == \A x \in Blocks : on[x] /= b

Init == /\ on   = [b \in Blocks |-> "table"]
        /\ hand = "empty"

\* Pickup b from table: b must be clear and on the table; hand empty.
Pickup(b) ==
    /\ hand = "empty"
    /\ on[b] = "table"
    /\ Clear(b)
    /\ on'   = [on EXCEPT ![b] = "hand"]
    /\ hand' = b

\* Putdown b onto the table: hand must hold b.
Putdown(b) ==
    /\ hand = b
    /\ on'   = [on EXCEPT ![b] = "table"]
    /\ hand' = "empty"

\* Unstack b from y: hand empty, b on y, b clear.
Unstack(b, y) ==
    /\ hand = "empty"
    /\ on[b] = y
    /\ y \in Blocks
    /\ Clear(b)
    /\ on'   = [on EXCEPT ![b] = "hand"]
    /\ hand' = b

\* Stack b on y: hand holds b, y is clear.
Stack(b, y) ==
    /\ hand = b
    /\ y \in Blocks
    /\ y /= b
    /\ Clear(y)
    /\ on'   = [on EXCEPT ![b] = y]
    /\ hand' = "empty"

Next ==
    \/ \E b \in Blocks : Pickup(b)
    \/ \E b \in Blocks : Putdown(b)
    \/ \E b \in Blocks, y \in Blocks : Unstack(b, y)
    \/ \E b \in Blocks, y \in Blocks : Stack(b, y)

Spec == Init /\ [][Next]_vars

\* Strong invariant:
\*   - At most one block in the hand.
\*   - "hand" appears in on iff hand /= "empty" and points to that block.
\*   - At most one block sits directly on any other block.
HeldCount  == Cardinality({ b \in Blocks : on[b] = "hand" })

SafetyInv ==
    /\ HeldCount <= 1
    /\ (hand = "empty") <=> (HeldCount = 0)
    /\ (hand /= "empty") => (on[hand] = "hand")
    /\ \A y \in Blocks :
          Cardinality({ b \in Blocks : on[b] = y }) <= 1

TypeOK == /\ on \in [Blocks -> Supports]
          /\ hand \in Blocks \cup {"empty"}
          /\ SafetyInv
====
