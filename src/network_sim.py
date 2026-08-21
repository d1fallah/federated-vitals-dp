"""
M3 connectivity-robustness — SimPy network-condition simulation
=====================================================================
Standalone (no torch/nvflare import) so the delivery model can be
calibrated and sanity-checked on its own before any FLARE integration.
Implements the roadmap's fixed Good/Moderate/Poor network profiles
(packet loss / latency / bandwidth / client dropout) and pre-registers,
per (client, round), whether that client's model update is delivered to
the FLARE server before a fixed round deadline.

Three independent gates per (client, round):
  1. dropout gate     -- Bernoulli(profile.dropout_rate): the ambulance is
                          off the network entirely this round.
  2. packet-loss gate -- Bernoulli(profile.packet_loss): the update message
                          is lost in transit. Applied directly as a
                          message-failure probability, NOT compounded
                          per-packet across some assumed packet count --
                          compounding (1-(1-loss)^n_packets) would inflate
                          "Good" (1% loss) to a much higher message-failure
                          rate for a ~62KB payload at typical MTU, which
                          contradicts the profile's own name.
  3. deadline gate    -- latency + payload/effective_bandwidth compared to
                          a fixed ROUND_DEADLINE_SEC. effective_bandwidth
                          divides the profile's bandwidth by the number of
                          clients simultaneously contending for it this
                          round (fair-share contention) -- a shared-resource
                          effect, so it is modeled with a real
                          simpy.Environment per round (one process per
                          gate-surviving client), not an independent
                          per-client draw.

A client that fails any gate does not deliver its update this round.
"""
import json
from dataclasses import dataclass

import numpy as np
import simpy


@dataclass(frozen=True)
class NetworkProfile:
    name: str
    packet_loss: float      # message-failure probability (fraction, e.g. 0.01 = 1%)
    latency_sec: float      # one-way latency, seconds
    bandwidth_bps: float    # bits per second
    dropout_rate: float     # fraction of (client, round) pairs offline entirely


# Fixed roadmap parameter table (SihhaDZ_Student_Roadmap_v8, M3 network
# profiles). These values are what makes the Good/Moderate/Poor ablation
# axis comparable across the paper -- do not change without a corresponding
# supervisor sign-off.
PROFILES = {
    "good": NetworkProfile("good", packet_loss=0.01, latency_sec=0.050, bandwidth_bps=10_000_000, dropout_rate=0.05),
    "moderate": NetworkProfile("moderate", packet_loss=0.05, latency_sec=0.200, bandwidth_bps=2_000_000, dropout_rate=0.20),
    "poor": NetworkProfile("poor", packet_loss=0.10, latency_sec=0.500, bandwidth_bps=500_000, dropout_rate=0.40),
}

DEFAULT_PAYLOAD_BYTES = 62_777  # measured directly from the real K=10 FL_global_model.pt checkpoint
ROUND_DEADLINE_SEC = 3.0


def simulate_round(profile, client_ids, payload_bytes, deadline_sec, rng):
    """Runs one round's delivery decisions inside a fresh simpy.Environment.
    Each client first rolls its dropout/packet-loss gates (independent,
    instantaneous); every gate-surviving client then contends for the
    profile's shared bandwidth (effective_bandwidth = bandwidth / n_active)
    via one SimPy process each, and is marked delivered only if
    latency + transfer_time <= deadline_sec.

    Returns {client_id: {"delivered": bool, "reason": str, "transfer_sec": float|None}}.
    """
    results = {}
    active = []
    for cid in client_ids:
        if rng.random() < profile.dropout_rate:
            results[cid] = {"delivered": False, "reason": "dropout", "transfer_sec": None}
            continue
        if rng.random() < profile.packet_loss:
            results[cid] = {"delivered": False, "reason": "packet_loss", "transfer_sec": None}
            continue
        active.append(cid)

    if not active:
        return results

    effective_bandwidth = profile.bandwidth_bps / len(active)
    transfer_sec = (payload_bytes * 8) / effective_bandwidth
    total_sec = profile.latency_sec + transfer_sec
    delivered = total_sec <= deadline_sec
    reason = "ok" if delivered else "deadline_exceeded"

    env = simpy.Environment()

    def client_proc(cid):
        yield env.timeout(total_sec)
        results[cid] = {"delivered": delivered, "reason": reason, "transfer_sec": round(transfer_sec, 4)}

    for cid in active:
        env.process(client_proc(cid))
    env.run()

    return results


def generate_schedule(profile_name, client_ids, num_rounds, seed=42,
                       payload_bytes=DEFAULT_PAYLOAD_BYTES, deadline_sec=ROUND_DEADLINE_SEC):
    """Pre-registers the full (client, round) delivery-outcome table before
    training starts, so the same schedule can be written to disk and re-read
    by every client process without needing shared RNG state across FLARE's
    per-site sandboxes.

    Enforces at least one delivery per round: if every client would
    otherwise fail delivery, the client with the shortest attempted transfer
    time (ties broken by client id) is force-delivered and tagged
    reason="forced_min_survivor_override". This is a simulation-infrastructure
    guard, not a network-modeling assumption: the non-DP FLARE job's
    ParamsType.FULL aggregation does a wholesale replace of the global model,
    so a round with zero deliveries would permanently null it (params={}) and
    crash the next round's strict=True load_state_dict. Real occurrence rate
    of this override is reported alongside the schedule so it can be checked
    to be rare (Good/Moderate) or at least disclosed (Poor), never silently
    relied on to produce the reported results.
    """
    profile = PROFILES[profile_name]
    rng = np.random.default_rng(seed)
    schedule = {}
    for r in range(num_rounds):
        results = simulate_round(profile, client_ids, payload_bytes, deadline_sec, rng)
        if not any(v["delivered"] for v in results.values()):
            fastest = min(client_ids, key=lambda cid: (results[cid]["transfer_sec"] is None, results[cid]["transfer_sec"] or float("inf"), cid))
            results[fastest] = {"delivered": True, "reason": "forced_min_survivor_override", "transfer_sec": results[fastest]["transfer_sec"]}
        schedule[str(r)] = {str(cid): results[cid] for cid in client_ids}
    return schedule


def write_schedule(schedule, path):
    with open(path, "w") as f:
        json.dump(schedule, f, indent=2)


def load_schedule(path):
    with open(path) as f:
        return json.load(f)


def participation_rate(schedule):
    """Fraction of (client, round) pairs delivered, from the schedule (a-priori, pre-training)."""
    total = sum(len(v) for v in schedule.values())
    delivered = sum(1 for v in schedule.values() for outcome in v.values() if outcome["delivered"])
    return delivered / total if total else 0.0


def breakdown_by_reason(schedule):
    counts = {}
    for v in schedule.values():
        for outcome in v.values():
            counts[outcome["reason"]] = counts.get(outcome["reason"], 0) + 1
    return counts


def per_round_participation(schedule):
    out = {}
    for r, v in schedule.items():
        out[r] = (sum(1 for outcome in v.values() if outcome["delivered"]) / len(v)) if v else 0.0
    return out


def _calibration_check():
    """Standalone sanity check (no FLARE/torch needed): prints the real
    delivery time per profile at full client-count contention (the worst-case
    bandwidth-sharing ceiling; actual per-round occupancy after dropout/loss
    gates is smaller, so real transfer times in generate_schedule() are
    faster than this ceiling) for K=10, so the three regimes' qualitative
    separation can be verified before wiring this into the federated job."""
    n_active = 10
    print(f"Calibration check: n_active={n_active}, payload={DEFAULT_PAYLOAD_BYTES} bytes, deadline={ROUND_DEADLINE_SEC}s\n")
    for name, profile in PROFILES.items():
        transfer_sec = (DEFAULT_PAYLOAD_BYTES * 8) / (profile.bandwidth_bps / n_active)
        total_sec = profile.latency_sec + transfer_sec
        verdict = "OK (within deadline)" if total_sec <= ROUND_DEADLINE_SEC else "MISSED (exceeds deadline)"
        print(f"  {name:10s} latency={profile.latency_sec:.3f}s  transfer={transfer_sec:.3f}s  "
              f"total={total_sec:.3f}s  -> {verdict}")

    print("\nMonte Carlo check (200 rounds, K=10, seed=42): a-priori participation rate and reason breakdown")
    for name in PROFILES:
        schedule = generate_schedule(name, list(range(10)), num_rounds=200, seed=42)
        rate = participation_rate(schedule)
        reasons = breakdown_by_reason(schedule)
        print(f"  {name:10s} participation_rate={rate:.4f}  reasons={reasons}")


if __name__ == "__main__":
    _calibration_check()
