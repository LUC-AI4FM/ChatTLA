---- MODULE UnboundedChannel ----
EXTENDS Naturals

CONSTANT Max

\* Bounded model of an unbounded channel: Send always succeeds (until Max for
\* finite checking); Recv blocks when empty.
\* sent     : total messages sent
\* received : total messages received
VARIABLES sent, received

vars == << sent, received >>

Init == /\ sent     = 0
        /\ received = 0

Send == /\ sent < Max
        /\ sent' = sent + 1
        /\ UNCHANGED received

Recv == /\ received < sent
        /\ received' = received + 1
        /\ UNCHANGED sent

\* Drain: when all in-flight messages have been received, reset counters
\* so the (finite) model can keep evolving.
Drain == /\ sent = Max
         /\ received = Max
         /\ sent' = 0
         /\ received' = 0

Next == \/ Send
        \/ Recv
        \/ Drain

Spec == Init /\ [][Next]_vars

\* Safety: pending messages = sent - received, must be >= 0 and <= Max.
ChannelInvariant == /\ received <= sent
                    /\ sent - received >= 0
                    /\ sent - received <= Max

TypeOK == /\ sent     \in 0..Max
          /\ received \in 0..Max
          /\ ChannelInvariant
====
