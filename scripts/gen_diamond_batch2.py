#!/usr/bin/env python3
"""Batch 2 of hand-crafted Diamond TLA+ specs."""
from __future__ import annotations
import json, logging, sys
from datetime import datetime
from pathlib import Path

log = logging.getLogger("gen_diamond_b2")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

_REPO_ROOT = Path(__file__).resolve().parents[1]
_DIAMOND_OUT = _REPO_ROOT / "data" / "processed" / "diamond_sft.jsonl"

SPECS: list[tuple[str, str, str]] = []

# ---------------------------------------------------------------------------
# 1. FIFO channel with bounded capacity and message IDs
# ---------------------------------------------------------------------------
SPECS.append(("manual2_fifo_channel",
"Write a TLA+ specification for a FIFO channel with bounded capacity. Messages have sequential IDs. The channel can send (enqueue) and receive (dequeue) messages. The channel length must never exceed its capacity.",
r"""---- MODULE FifoChannel ----
EXTENDS Integers
CONSTANT Capacity
VARIABLES chanLen, nextId

Init == chanLen = 0 /\ nextId = 1

Send == chanLen < Capacity /\ nextId < Capacity * 2
        /\ chanLen' = chanLen + 1 /\ nextId' = nextId + 1

Receive == chanLen > 0
           /\ chanLen' = chanLen - 1 /\ UNCHANGED nextId

Next == Send \/ Receive \/ UNCHANGED <<chanLen, nextId>>

Spec == Init /\ [][Next]_<<chanLen, nextId>>

TypeOK == chanLen \in 0..Capacity /\ nextId \in 1..(Capacity * 2)

NoOverflow == chanLen <= Capacity

SafetyInv == chanLen >= 0 /\ chanLen <= Capacity
====
"""))

# ---------------------------------------------------------------------------
# 2. Resource pool (checkout/return resources, track available count)
# ---------------------------------------------------------------------------
SPECS.append(("manual2_resource_pool",
"Write a TLA+ specification for a resource pool. Resources can be checked out and returned. Track the number of available and in-use resources. The total must remain constant.",
r"""---- MODULE ResourcePool ----
EXTENDS Integers
CONSTANT N
VARIABLES available, inUse

Init == available = N /\ inUse = 0

Checkout == available > 0
            /\ available' = available - 1 /\ inUse' = inUse + 1

Return == inUse > 0
          /\ inUse' = inUse - 1 /\ available' = available + 1

Next == Checkout \/ Return \/ UNCHANGED <<available, inUse>>

Spec == Init /\ [][Next]_<<available, inUse>>

TypeOK == available \in 0..N /\ inUse \in 0..N

ResourceConserved == available + inUse = N

SafetyInv == available >= 0 /\ inUse >= 0
====
"""))

# ---------------------------------------------------------------------------
# 3. Circuit breaker pattern (closed/open/half-open with failure count)
# ---------------------------------------------------------------------------
SPECS.append(("manual2_circuit_breaker",
"Write a TLA+ specification for a circuit breaker pattern with three states: Closed, Open, and HalfOpen. Track consecutive failures. When failures reach a threshold, the breaker opens. In HalfOpen, a single success closes it or a failure reopens it.",
r"""---- MODULE CircuitBreaker ----
EXTENDS Integers
CONSTANT Threshold
VARIABLES state, failures

Init == state = "Closed" /\ failures = 0

Success == state = "Closed"
           /\ failures' = 0 /\ UNCHANGED state

Failure == state = "Closed" /\ failures < Threshold
           /\ failures' = failures + 1
           /\ state' = IF failures + 1 >= Threshold THEN "Open" ELSE "Closed"

Trip == state = "Open"
        /\ state' = "HalfOpen" /\ UNCHANGED failures

HalfSuccess == state = "HalfOpen"
               /\ state' = "Closed" /\ failures' = 0

HalfFailure == state = "HalfOpen"
               /\ state' = "Open" /\ UNCHANGED failures

Next == Success \/ Failure \/ Trip \/ HalfSuccess \/ HalfFailure
        \/ UNCHANGED <<state, failures>>

Spec == Init /\ [][Next]_<<state, failures>>

TypeOK == state \in {"Closed", "Open", "HalfOpen"}
          /\ failures \in 0..Threshold

FailuresBounded == failures <= Threshold

SafetyInv == state = "Open" => failures >= Threshold
====
"""))

# ---------------------------------------------------------------------------
# 4. Two-phase commit protocol (coordinator + 2 participants)
# ---------------------------------------------------------------------------
SPECS.append(("manual2_two_phase_commit",
"Write a TLA+ specification for a two-phase commit protocol with a coordinator and two participants. The coordinator sends prepare, participants vote, and the coordinator decides commit or abort. All participants must agree for commit.",
r"""---- MODULE TwoPhaseCommit ----
EXTENDS Integers
VARIABLES coordState, p1Vote, p2Vote
vars == <<coordState, p1Vote, p2Vote>>

Init == coordState = "init" /\ p1Vote = "none" /\ p2Vote = "none"

Prepare == /\ coordState = "init"
           /\ coordState' = "waiting"
           /\ p1Vote' = p1Vote
           /\ p2Vote' = p2Vote

P1VoteYes == /\ coordState = "waiting" /\ p1Vote = "none"
             /\ p1Vote' = "yes"
             /\ coordState' = coordState
             /\ p2Vote' = p2Vote

P1VoteNo == /\ coordState = "waiting" /\ p1Vote = "none"
            /\ p1Vote' = "no"
            /\ coordState' = coordState
            /\ p2Vote' = p2Vote

P2VoteYes == /\ coordState = "waiting" /\ p2Vote = "none"
             /\ p2Vote' = "yes"
             /\ coordState' = coordState
             /\ p1Vote' = p1Vote

P2VoteNo == /\ coordState = "waiting" /\ p2Vote = "none"
            /\ p2Vote' = "no"
            /\ coordState' = coordState
            /\ p1Vote' = p1Vote

DecideCommit == /\ coordState = "waiting"
                /\ p1Vote = "yes" /\ p2Vote = "yes"
                /\ coordState' = "committed"
                /\ p1Vote' = p1Vote
                /\ p2Vote' = p2Vote

DecideAbort == /\ coordState = "waiting"
               /\ p1Vote /= "none" /\ p2Vote /= "none"
               /\ ~(p1Vote = "yes" /\ p2Vote = "yes")
               /\ coordState' = "aborted"
               /\ p1Vote' = p1Vote
               /\ p2Vote' = p2Vote

Next == Prepare \/ P1VoteYes \/ P1VoteNo \/ P2VoteYes \/ P2VoteNo
        \/ DecideCommit \/ DecideAbort
        \/ UNCHANGED vars

Spec == Init /\ [][Next]_vars

TypeOK == coordState \in {"init", "waiting", "committed", "aborted"}
          /\ p1Vote \in {"none", "yes", "no"}
          /\ p2Vote \in {"none", "yes", "no"}

SafetyInv == coordState = "committed" => (p1Vote = "yes" /\ p2Vote = "yes")
====
"""))

# ---------------------------------------------------------------------------
# 5. Simple key-value store with bounded keys
# ---------------------------------------------------------------------------
SPECS.append(("manual2_kv_store",
"Write a TLA+ specification for a simple key-value store with a bounded number of keys. Keys can be written and deleted. Track the count of stored keys. The store must not exceed its capacity.",
r"""---- MODULE KvStore ----
EXTENDS Integers
CONSTANT Capacity
VARIABLES keyCount, lastOp

Init == keyCount = 0 /\ lastOp = "none"

Write == keyCount < Capacity
         /\ keyCount' = keyCount + 1 /\ lastOp' = "write"

Delete == keyCount > 0
          /\ keyCount' = keyCount - 1 /\ lastOp' = "delete"

Read == keyCount > 0
        /\ UNCHANGED keyCount /\ lastOp' = "read"

Next == Write \/ Delete \/ Read \/ UNCHANGED <<keyCount, lastOp>>

Spec == Init /\ [][Next]_<<keyCount, lastOp>>

TypeOK == keyCount \in 0..Capacity
          /\ lastOp \in {"none", "write", "delete", "read"}

NoOverflow == keyCount <= Capacity

SafetyInv == keyCount >= 0 /\ keyCount <= Capacity
====
"""))

# ---------------------------------------------------------------------------
# 6. Rate limiter (token bucket with refill)
# ---------------------------------------------------------------------------
SPECS.append(("manual2_rate_limiter",
"Write a TLA+ specification for a token bucket rate limiter. Tokens are consumed by requests and periodically refilled. The bucket has a maximum capacity. Requests are only allowed when tokens are available.",
r"""---- MODULE RateLimiter ----
EXTENDS Integers
CONSTANT Max
VARIABLES tokens, requests

Init == tokens = Max /\ requests = 0

Consume == tokens > 0 /\ requests < Max * 2
           /\ tokens' = tokens - 1 /\ requests' = requests + 1

Refill == tokens < Max
          /\ tokens' = tokens + 1 /\ UNCHANGED requests

Next == Consume \/ Refill \/ UNCHANGED <<tokens, requests>>

Spec == Init /\ [][Next]_<<tokens, requests>>

TypeOK == tokens \in 0..Max /\ requests \in 0..(Max * 2)

TokensBounded == tokens >= 0 /\ tokens <= Max

SafetyInv == tokens >= 0
====
"""))

# ---------------------------------------------------------------------------
# 7. Job scheduler (submit/run/complete with max concurrent)
# ---------------------------------------------------------------------------
SPECS.append(("manual2_job_scheduler",
"Write a TLA+ specification for a job scheduler. Jobs can be submitted to a queue, started (up to a max concurrency), and completed. Track pending, running, and done counts.",
r"""---- MODULE JobScheduler ----
EXTENDS Integers
CONSTANT Max
VARIABLES pending, running, done

Init == pending = 0 /\ running = 0 /\ done = 0

Submit == pending + running + done < Max * 2
          /\ pending' = pending + 1 /\ UNCHANGED <<running, done>>

Start == pending > 0 /\ running < Max
         /\ pending' = pending - 1 /\ running' = running + 1
         /\ UNCHANGED done

Complete == running > 0
            /\ running' = running - 1 /\ done' = done + 1
            /\ UNCHANGED pending

Next == Submit \/ Start \/ Complete
        \/ UNCHANGED <<pending, running, done>>

Spec == Init /\ [][Next]_<<pending, running, done>>

TypeOK == pending \in 0..(Max * 2)
          /\ running \in 0..Max
          /\ done \in 0..(Max * 2)

AtMostMaxRunning == running <= Max

SafetyInv == running >= 0 /\ pending >= 0 /\ done >= 0
====
"""))

# ---------------------------------------------------------------------------
# 8. Distributed lock (acquire/release with TTL counter)
# ---------------------------------------------------------------------------
SPECS.append(("manual2_dist_lock",
"Write a TLA+ specification for a distributed lock with a TTL (time-to-live) counter. A lock can be acquired, held (TTL decrements), and released. When TTL expires, the lock is automatically released.",
r"""---- MODULE DistLock ----
EXTENDS Integers
CONSTANT MaxTTL
VARIABLES locked, ttl

Init == locked = FALSE /\ ttl = 0

Acquire == locked = FALSE
           /\ locked' = TRUE /\ ttl' = MaxTTL

TickTTL == locked = TRUE /\ ttl > 1
           /\ ttl' = ttl - 1 /\ UNCHANGED locked

Expire == locked = TRUE /\ ttl <= 1
          /\ locked' = FALSE /\ ttl' = 0

Release == locked = TRUE
           /\ locked' = FALSE /\ ttl' = 0

Next == Acquire \/ TickTTL \/ Expire \/ Release
        \/ UNCHANGED <<locked, ttl>>

Spec == Init /\ [][Next]_<<locked, ttl>>

TypeOK == locked \in BOOLEAN /\ ttl \in 0..MaxTTL

SafetyInv == locked = FALSE => ttl = 0

TTLBounded == ttl >= 0 /\ ttl <= MaxTTL
====
"""))

# ---------------------------------------------------------------------------
# 9. File system permissions (read/write/execute with user roles)
# ---------------------------------------------------------------------------
SPECS.append(("manual2_file_perms",
"Write a TLA+ specification for a file permission system. A file has permission bits for read, write, and execute. An admin can grant or revoke permissions. Write requires read. Track access attempts.",
r"""---- MODULE FilePerms ----
EXTENDS Integers
VARIABLES canRead, canWrite, canExec, accessCount
CONSTANT Max

Init == canRead = FALSE /\ canWrite = FALSE /\ canExec = FALSE
        /\ accessCount = 0

GrantRead == canRead = FALSE
             /\ canRead' = TRUE /\ UNCHANGED <<canWrite, canExec, accessCount>>

RevokeRead == canRead = TRUE /\ canWrite = FALSE
              /\ canRead' = FALSE /\ UNCHANGED <<canWrite, canExec, accessCount>>

GrantWrite == canWrite = FALSE /\ canRead = TRUE
              /\ canWrite' = TRUE /\ UNCHANGED <<canRead, canExec, accessCount>>

RevokeWrite == canWrite = TRUE
               /\ canWrite' = FALSE /\ UNCHANGED <<canRead, canExec, accessCount>>

GrantExec == canExec = FALSE
             /\ canExec' = TRUE /\ UNCHANGED <<canRead, canWrite, accessCount>>

RevokeExec == canExec = TRUE
              /\ canExec' = FALSE /\ UNCHANGED <<canRead, canWrite, accessCount>>

Access == accessCount < Max /\ (canRead \/ canWrite \/ canExec)
          /\ accessCount' = accessCount + 1
          /\ UNCHANGED <<canRead, canWrite, canExec>>

Next == GrantRead \/ RevokeRead \/ GrantWrite \/ RevokeWrite
        \/ GrantExec \/ RevokeExec \/ Access
        \/ UNCHANGED <<canRead, canWrite, canExec, accessCount>>

Spec == Init /\ [][Next]_<<canRead, canWrite, canExec, accessCount>>

TypeOK == canRead \in BOOLEAN /\ canWrite \in BOOLEAN
          /\ canExec \in BOOLEAN /\ accessCount \in 0..Max

NoWriteWithoutRead == canWrite = TRUE => canRead = TRUE

SafetyInv == accessCount <= Max
====
"""))

# ---------------------------------------------------------------------------
# 10. Cache with eviction (bounded entries, LRU approximation)
# ---------------------------------------------------------------------------
SPECS.append(("manual2_cache_eviction",
"Write a TLA+ specification for a cache with bounded capacity and eviction. Entries can be added and evicted. Track the number of entries and total hits/misses. The cache size must not exceed capacity.",
r"""---- MODULE CacheEviction ----
EXTENDS Integers
CONSTANT Capacity
VARIABLES entries, hits, misses

Init == entries = 0 /\ hits = 0 /\ misses = 0

CacheHit == entries > 0 /\ hits + misses < Capacity * 3
            /\ hits' = hits + 1 /\ UNCHANGED <<entries, misses>>

CacheMiss == entries < Capacity /\ hits + misses < Capacity * 3
             /\ entries' = entries + 1 /\ misses' = misses + 1
             /\ UNCHANGED hits

Evict == entries > 0
         /\ entries' = entries - 1 /\ UNCHANGED <<hits, misses>>

MissEvict == entries >= Capacity /\ hits + misses < Capacity * 3
             /\ misses' = misses + 1 /\ UNCHANGED <<entries, hits>>

Next == CacheHit \/ CacheMiss \/ Evict \/ MissEvict
        \/ UNCHANGED <<entries, hits, misses>>

Spec == Init /\ [][Next]_<<entries, hits, misses>>

TypeOK == entries \in 0..Capacity
          /\ hits \in 0..(Capacity * 3)
          /\ misses \in 0..(Capacity * 3)

NoOverflow == entries <= Capacity

SafetyInv == entries >= 0 /\ entries <= Capacity
====
"""))

# ---------------------------------------------------------------------------
# 11. Consensus voting (N voters, majority decision)
# ---------------------------------------------------------------------------
SPECS.append(("manual2_consensus_vote",
"Write a TLA+ specification for a consensus voting protocol. Voters cast yes or no votes. Once all have voted, the result is decided by majority. Track votes cast, yes count, and decision status.",
r"""---- MODULE ConsensusVote ----
EXTENDS Integers
CONSTANT N
VARIABLES yesVotes, noVotes, decided
vars == <<yesVotes, noVotes, decided>>

Init == yesVotes = 0 /\ noVotes = 0 /\ decided = "pending"

VoteYes == /\ decided = "pending" /\ yesVotes + noVotes < N
           /\ yesVotes' = yesVotes + 1
           /\ noVotes' = noVotes
           /\ decided' = decided

VoteNo == /\ decided = "pending" /\ yesVotes + noVotes < N
          /\ noVotes' = noVotes + 1
          /\ yesVotes' = yesVotes
          /\ decided' = decided

DecideAccept == /\ decided = "pending" /\ yesVotes + noVotes = N
                /\ yesVotes * 2 > N
                /\ decided' = "accepted"
                /\ yesVotes' = yesVotes
                /\ noVotes' = noVotes

DecideReject == /\ decided = "pending" /\ yesVotes + noVotes = N
                /\ ~(yesVotes * 2 > N)
                /\ decided' = "rejected"
                /\ yesVotes' = yesVotes
                /\ noVotes' = noVotes

Next == VoteYes \/ VoteNo \/ DecideAccept \/ DecideReject
        \/ UNCHANGED vars

Spec == Init /\ [][Next]_vars

TypeOK == yesVotes \in 0..N /\ noVotes \in 0..N
          /\ decided \in {"pending", "accepted", "rejected"}

VotesBounded == yesVotes + noVotes <= N

SafetyInv == decided = "accepted" => yesVotes * 2 > N
====
"""))

# ---------------------------------------------------------------------------
# 12. Pipeline stages (3-stage pipeline with items flowing through)
# ---------------------------------------------------------------------------
SPECS.append(("manual2_pipeline",
"Write a TLA+ specification for a 3-stage processing pipeline. Items enter stage 1, move to stage 2, then stage 3, and finally exit. Each stage can hold at most one item. Track items in each stage.",
r"""---- MODULE PipelineStages ----
EXTENDS Integers
CONSTANT Max
VARIABLES s1, s2, s3, produced

Init == s1 = FALSE /\ s2 = FALSE /\ s3 = FALSE /\ produced = 0

Enter == s1 = FALSE /\ produced < Max
         /\ s1' = TRUE /\ produced' = produced + 1
         /\ UNCHANGED <<s2, s3>>

Advance12 == s1 = TRUE /\ s2 = FALSE
             /\ s1' = FALSE /\ s2' = TRUE /\ UNCHANGED <<s3, produced>>

Advance23 == s2 = TRUE /\ s3 = FALSE
             /\ s2' = FALSE /\ s3' = TRUE /\ UNCHANGED <<s1, produced>>

Exit == s3 = TRUE
        /\ s3' = FALSE /\ UNCHANGED <<s1, s2, produced>>

Next == Enter \/ Advance12 \/ Advance23 \/ Exit
        \/ UNCHANGED <<s1, s2, s3, produced>>

Spec == Init /\ [][Next]_<<s1, s2, s3, produced>>

TypeOK == s1 \in BOOLEAN /\ s2 \in BOOLEAN /\ s3 \in BOOLEAN
          /\ produced \in 0..Max

SafetyInv == produced <= Max

ProducedBounded == produced >= 0 /\ produced <= Max
====
"""))

# ---------------------------------------------------------------------------
# 13. Temperature controller (heating/cooling with bounds)
# ---------------------------------------------------------------------------
SPECS.append(("manual2_temp_controller",
"Write a TLA+ specification for a temperature controller. The controller heats or cools to keep temperature within bounds. Track temperature and mode (heating/cooling/idle). Temperature must stay in range.",
r"""---- MODULE TempController ----
EXTENDS Integers
CONSTANT Max
VARIABLES temp, mode

Init == temp = Max \div 2 /\ mode = "idle"

Heat == temp < Max
        /\ temp' = temp + 1
        /\ mode' = "heating"

Cool == temp > 0
        /\ temp' = temp - 1
        /\ mode' = "cooling"

Idle == mode' = "idle" /\ UNCHANGED temp

Next == Heat \/ Cool \/ Idle
        \/ UNCHANGED <<temp, mode>>

Spec == Init /\ [][Next]_<<temp, mode>>

TypeOK == temp \in 0..Max /\ mode \in {"heating", "cooling", "idle"}

TempBounded == temp >= 0 /\ temp <= Max

SafetyInv == temp >= 0 /\ temp <= Max
====
"""))

# ---------------------------------------------------------------------------
# 14. Auction system (bidding with minimum increment)
# ---------------------------------------------------------------------------
SPECS.append(("manual2_auction",
"Write a TLA+ specification for an auction system. Bidders place bids that must exceed the current highest bid. The auction can be open or closed. Track current bid and bid count.",
r"""---- MODULE AuctionSystem ----
EXTENDS Integers
CONSTANT Max
VARIABLES currentBid, bidCount, auctionOpen

Init == currentBid = 0 /\ bidCount = 0 /\ auctionOpen = TRUE

PlaceBid == auctionOpen = TRUE /\ currentBid < Max /\ bidCount < Max
            /\ currentBid' = currentBid + 1
            /\ bidCount' = bidCount + 1
            /\ UNCHANGED auctionOpen

CloseAuction == auctionOpen = TRUE /\ bidCount > 0
                /\ auctionOpen' = FALSE
                /\ UNCHANGED <<currentBid, bidCount>>

Next == PlaceBid \/ CloseAuction
        \/ UNCHANGED <<currentBid, bidCount, auctionOpen>>

Spec == Init /\ [][Next]_<<currentBid, bidCount, auctionOpen>>

TypeOK == currentBid \in 0..Max /\ bidCount \in 0..Max
          /\ auctionOpen \in BOOLEAN

BidsBounded == bidCount <= Max /\ currentBid <= Max

SafetyInv == auctionOpen = FALSE => bidCount > 0
====
"""))

# ---------------------------------------------------------------------------
# 15. Database transaction (begin/commit/rollback with isolation)
# ---------------------------------------------------------------------------
SPECS.append(("manual2_db_transaction",
"Write a TLA+ specification for database transactions. A transaction can begin, perform reads/writes, and then commit or rollback. Track transaction state and the number of operations performed. Ensure operations only happen in active transactions.",
r"""---- MODULE DbTransaction ----
EXTENDS Integers
CONSTANT Max
VARIABLES txState, opCount, committed

Init == txState = "idle" /\ opCount = 0 /\ committed = 0

BeginTx == txState = "idle"
           /\ txState' = "active" /\ UNCHANGED <<opCount, committed>>

DoOp == txState = "active" /\ opCount < Max
        /\ opCount' = opCount + 1 /\ UNCHANGED <<txState, committed>>

CommitTx == txState = "active" /\ committed < Max
            /\ txState' = "idle" /\ committed' = committed + 1
            /\ opCount' = 0

RollbackTx == txState = "active"
              /\ txState' = "idle" /\ opCount' = 0
              /\ UNCHANGED committed

Next == BeginTx \/ DoOp \/ CommitTx \/ RollbackTx
        \/ UNCHANGED <<txState, opCount, committed>>

Spec == Init /\ [][Next]_<<txState, opCount, committed>>

TypeOK == txState \in {"idle", "active"}
          /\ opCount \in 0..Max
          /\ committed \in 0..Max

SafetyInv == txState = "idle" => opCount = 0

OpsBounded == opCount <= Max /\ committed <= Max
====
"""))


def validate_and_save(do_validate=True, do_save=False):
    from scripts.diamond_sft_gen import validate_diamond, save_diamond_sft
    results = []
    for pid, prompt_text, spec in SPECS:
        if do_validate:
            r = validate_diamond(spec, prompt_id=pid, prompt_text=prompt_text, model="manual")
            results.append(r)
            status = "DIAMOND" if r.is_diamond else f"FAIL({r.error[:60]})"
            log.info(f"  {pid}: {status} states={r.distinct_states} inv={r.invariants_checked} mut={r.mutation_caught}")
        else:
            results.append(None)
    diamond_count = sum(1 for r in results if r and r.is_diamond)
    log.info(f"\nResults: {diamond_count}/{len(SPECS)} Diamond")
    if do_save and diamond_count > 0:
        diamond_results = [r for r in results if r and r.is_diamond]
        save_diamond_sft(diamond_results, append=True)
    return results


if __name__ == "__main__":
    do_save = "--save" in sys.argv
    validate_and_save(do_validate=True, do_save=do_save)
