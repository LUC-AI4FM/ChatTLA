---- MODULE TcpClose ----
(***************************************************************************)
(* TCP four-way connection-termination handshake.                          *)
(* Each endpoint walks through                                             *)
(*   established -> fin_wait -> closing -> time_wait -> closed             *)
(* on the active-close side, and                                           *)
(*   established -> close_wait -> last_ack -> closed                       *)
(* on the passive-close side.                                              *)
(* Strong safety: an endpoint reaches "closed" only along legal            *)
(* transitions, and the active closer always reaches time_wait first.      *)
(***************************************************************************)
EXTENDS Naturals

VARIABLES aState, pState, finA, finP, ackA, ackP

vars == << aState, pState, finA, finP, ackA, ackP >>

ActiveStates  == {"established", "fin_wait", "closing", "time_wait", "closed"}
PassiveStates == {"established", "close_wait", "last_ack", "closed"}

Init ==
    /\ aState = "established"
    /\ pState = "established"
    /\ finA = FALSE      \* FIN sent by active side, in flight
    /\ finP = FALSE      \* FIN sent by passive side, in flight
    /\ ackA = FALSE      \* ACK from active side acknowledging passive's FIN
    /\ ackP = FALSE      \* ACK from passive side acknowledging active's FIN

\* 1. Active side initiates close: send FIN, enter fin_wait.
ActiveSendFin ==
    /\ aState = "established"
    /\ aState' = "fin_wait"
    /\ finA' = TRUE
    /\ UNCHANGED << pState, finP, ackA, ackP >>

\* 2. Passive side receives FIN, ACKs it, enters close_wait.
PassiveRecvFin ==
    /\ pState = "established"
    /\ finA = TRUE
    /\ pState' = "close_wait"
    /\ ackP' = TRUE
    /\ finA' = FALSE
    /\ UNCHANGED << aState, finP, ackA >>

\* 3. Active side receives the ACK, transitions fin_wait -> closing.
ActiveRecvAck ==
    /\ aState = "fin_wait"
    /\ ackP = TRUE
    /\ aState' = "closing"
    /\ ackP' = FALSE
    /\ UNCHANGED << pState, finA, finP, ackA >>

\* 4. Passive side sends its own FIN, enters last_ack.
PassiveSendFin ==
    /\ pState = "close_wait"
    /\ pState' = "last_ack"
    /\ finP' = TRUE
    /\ UNCHANGED << aState, finA, ackA, ackP >>

\* 5. Active side receives the FIN, ACKs it, enters time_wait.
ActiveRecvFin ==
    /\ aState = "closing"
    /\ finP = TRUE
    /\ aState' = "time_wait"
    /\ ackA' = TRUE
    /\ finP' = FALSE
    /\ UNCHANGED << pState, finA, ackP >>

\* 6. Passive side receives final ACK, enters closed.
PassiveRecvAck ==
    /\ pState = "last_ack"
    /\ ackA = TRUE
    /\ pState' = "closed"
    /\ ackA' = FALSE
    /\ UNCHANGED << aState, finA, finP, ackP >>

\* 7. Active side eventually leaves time_wait.
ActiveTimeout ==
    /\ aState = "time_wait"
    /\ aState' = "closed"
    /\ UNCHANGED << pState, finA, finP, ackA, ackP >>

Done ==
    /\ aState = "closed"
    /\ pState = "closed"
    /\ UNCHANGED vars

Next ==
    \/ ActiveSendFin \/ PassiveRecvFin \/ ActiveRecvAck \/ PassiveSendFin
    \/ ActiveRecvFin \/ PassiveRecvAck \/ ActiveTimeout \/ Done

Spec == Init /\ [][Next]_vars

\* Strong safety conjoined into TypeOK.
TypeOK ==
    /\ aState \in ActiveStates
    /\ pState \in PassiveStates
    /\ finA \in BOOLEAN /\ finP \in BOOLEAN
    /\ ackA \in BOOLEAN /\ ackP \in BOOLEAN
    \* Active closed implies passive closed or last_ack.
    /\ (aState = "closed") => (pState \in {"last_ack", "closed"})
    \* Passive cannot have closed before active sent FIN.
    /\ (pState # "established") => (aState # "established")
    \* Passive last_ack means active has already initiated close.
    /\ (pState = "last_ack") => (aState \in {"fin_wait", "closing", "time_wait", "closed"})
    \* Only the active side can be in time_wait.
    /\ (aState = "time_wait") => (pState \in {"last_ack", "closed"})
====
