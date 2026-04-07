---- MODULE TcpHandshake ----
(***************************************************************************)
(* TCP three-way handshake between a client and a server.                  *)
(* States per endpoint: closed, listen, syn_sent, syn_recv, established.   *)
(* Messages in flight: SYN, SYN-ACK, ACK.                                  *)
(* Strong safety: a peer reaches "established" only after seeing the       *)
(* matching message from the other side; both peers agree on the           *)
(* connection.                                                             *)
(***************************************************************************)
EXTENDS Naturals

States == {"closed", "listen", "syn_sent", "syn_recv", "established"}
Msgs   == {"SYN", "SYNACK", "ACK"}

VARIABLES cState, sState, channel

vars == << cState, sState, channel >>

Init ==
    /\ cState = "closed"
    /\ sState = "listen"
    /\ channel = {}

\* Client opens, sends SYN, becomes SYN_SENT.
ClientOpen ==
    /\ cState = "closed"
    /\ cState' = "syn_sent"
    /\ channel' = channel \cup {"SYN"}
    /\ UNCHANGED sState

\* Server receives SYN, replies with SYN-ACK, becomes SYN_RECV.
ServerRecvSyn ==
    /\ sState = "listen"
    /\ "SYN" \in channel
    /\ sState' = "syn_recv"
    /\ channel' = (channel \ {"SYN"}) \cup {"SYNACK"}
    /\ UNCHANGED cState

\* Client receives SYN-ACK, replies with ACK, becomes ESTABLISHED.
ClientRecvSynAck ==
    /\ cState = "syn_sent"
    /\ "SYNACK" \in channel
    /\ cState' = "established"
    /\ channel' = (channel \ {"SYNACK"}) \cup {"ACK"}
    /\ UNCHANGED sState

\* Server receives ACK, becomes ESTABLISHED.
ServerRecvAck ==
    /\ sState = "syn_recv"
    /\ "ACK" \in channel
    /\ sState' = "established"
    /\ channel' = channel \ {"ACK"}
    /\ UNCHANGED cState

Done ==
    /\ cState = "established"
    /\ sState = "established"
    /\ channel = {}
    /\ UNCHANGED vars

Next ==
    \/ ClientOpen \/ ServerRecvSyn \/ ClientRecvSynAck \/ ServerRecvAck \/ Done

Spec == Init /\ [][Next]_vars

\* Strong safety conjoined into TypeOK: the only way the client is in
\* "established" is if it has already produced an ACK (which the server
\* will or has consumed); and the server can only be "established" after
\* the client.  This rules out unilateral establishment.
TypeOK ==
    /\ cState \in States
    /\ sState \in States
    /\ channel \subseteq Msgs
    \* Client invariant: never in listen state.
    /\ cState # "listen"
    \* Server invariant: never in syn_sent state.
    /\ sState # "syn_sent"
    \* Server can only reach syn_recv after the client opened.
    /\ (sState = "syn_recv") => (cState \in {"syn_sent", "established"})
    \* Server established implies client established.
    /\ (sState = "established") => (cState = "established")
    \* Client established implies server already saw the SYN.
    /\ (cState = "established") => (sState \in {"syn_recv", "established"})
====
