---- MODULE OptimisticLockVersion ----
(***************************************************************************)
(*  Compare-and-swap on a versioned record.  Each writer reads the       *)
(*  current version, computes a new value, then attempts to commit by   *)
(*  CASing the version field.  If a concurrent writer has bumped the    *)
(*  version in the meantime the CAS fails and the write is lost.        *)
(*                                                                         *)
(*  Strong invariant: the version is monotone -- it never decreases --  *)
(*  and a successful write strictly increases it.                       *)
(***************************************************************************)
EXTENDS Naturals

CONSTANTS Writers, MaxVersion

VARIABLES version, snap, status

vars == << version, snap, status >>

States == {"idle", "reading", "committed", "aborted"}

Init == /\ version = 0
        /\ snap    = [w \in Writers |-> 0]
        /\ status  = [w \in Writers |-> "idle"]

\* Read the version field into a local snapshot.
Read(w) == /\ status[w] = "idle"
           /\ snap'   = [snap   EXCEPT ![w] = version]
           /\ status' = [status EXCEPT ![w] = "reading"]
           /\ UNCHANGED version

\* Successful CAS: snap matches the current version.
CommitOk(w) == /\ status[w] = "reading"
               /\ snap[w] = version
               /\ version < MaxVersion
               /\ version' = version + 1
               /\ status'  = [status EXCEPT ![w] = "committed"]
               /\ UNCHANGED snap

\* Failed CAS: someone else won the race.
CommitFail(w) == /\ status[w] = "reading"
                 /\ snap[w] # version
                 /\ status' = [status EXCEPT ![w] = "aborted"]
                 /\ UNCHANGED << version, snap >>

Next == \/ \E w \in Writers : Read(w)
        \/ \E w \in Writers : CommitOk(w)
        \/ \E w \in Writers : CommitFail(w)

Spec == Init /\ [][Next]_vars /\ WF_vars(Next)

\* Strong invariant: monotone version, and any committed writer's
\* recorded snapshot is strictly less than the current version.
TypeOK == /\ version \in 0..MaxVersion
          /\ snap    \in [Writers -> 0..MaxVersion]
          /\ status  \in [Writers -> States]
          /\ \A w \in Writers : snap[w] <= version
          /\ \A w \in Writers :
                status[w] = "committed" => snap[w] < version
====
