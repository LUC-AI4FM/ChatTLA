---- MODULE Saga ----
(***************************************************************************)
(*  Long-running Saga: a sequence of forward steps each with a           *)
(*  compensating action.  If step k fails, every step j with j < k that  *)
(*  has been executed is compensated in reverse order.                   *)
(*                                                                         *)
(*  Strong invariant: either the saga is on its forward path with steps  *)
(*  1..done committed, or it is on its compensation path with steps      *)
(*  1..done committed and steps undone..done compensated -- i.e. for    *)
(*  every executed step we either have a downstream commit or a          *)
(*  compensation, never both.                                            *)
(***************************************************************************)
EXTENDS Naturals

CONSTANTS NumSteps

Steps == 1..NumSteps

VARIABLES done, undone, status

vars == << done, undone, status >>

States == {"forward", "compensating", "completed", "rolled_back"}

Init == /\ done   = 0
        /\ undone = 0
        /\ status = "forward"

\* Execute the next forward step.
Forward == /\ status = "forward"
           /\ done < NumSteps
           /\ done' = done + 1
           /\ UNCHANGED << undone, status >>

\* The whole saga commits successfully.
Complete == /\ status = "forward"
            /\ done = NumSteps
            /\ status' = "completed"
            /\ UNCHANGED << done, undone >>

\* A forward step fails: switch to compensation, starting with the
\* most-recently committed step.
Fail == /\ status = "forward"
        /\ done > 0
        /\ done < NumSteps
        /\ status' = "compensating"
        /\ undone' = done
        /\ UNCHANGED done

\* Run the next compensating action in reverse.
Compensate == /\ status = "compensating"
              /\ undone > 0
              /\ undone' = undone - 1
              /\ UNCHANGED << done, status >>

Rollback == /\ status = "compensating"
            /\ undone = 0
            /\ status' = "rolled_back"
            /\ UNCHANGED << done, undone >>

Next == \/ Forward \/ Complete \/ Fail \/ Compensate \/ Rollback

Spec == Init /\ [][Next]_vars /\ WF_vars(Next)

\* Strong invariant: undone <= done; on completion, no compensation owed;
\* on rollback, every executed step has been undone.
TypeOK == /\ done   \in 0..NumSteps
          /\ undone \in 0..NumSteps
          /\ status \in States
          /\ undone <= done
          /\ (status = "forward")      => (undone = 0)
          /\ (status = "completed")    => (done = NumSteps /\ undone = 0)
          /\ (status = "rolled_back")  => (undone = 0)
====
