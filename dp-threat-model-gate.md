# Module 3 (M3) — DP Threat-Model Gate (v2 — revised per supervisor decision, 17 Aug 2026)

Roadmap requirement (SihhaDZ Roadmap v8, M3 §2): this gate must be defined,
documented, and supervisor-approved *before* any federated or
differentially-private training begins. The centralized-only and non-DP
federated baselines are explicitly outside this gate's scope per the
supervisor's decision of 17 Aug 2026 and may proceed independently of it.

**Revision note:** v1 of this document was rejected on two points: (1) it
claimed client-level privacy "falls out of" standard local Opacus DP-SGD,
which is incorrect — Opacus' per-example clipping/noising is a record-level
mechanism by construction, not a client-level one; (2) δ = 0.05 was proposed
without project-specific justification. Both are corrected below.

## 1. Privacy Unit and Mechanism — Option Chosen: A (Record-level DP)

Per the supervisor's decision, two valid options were offered:

| Option | Privacy Unit | Mechanism |
|---|---|---|
| A. Record-level DP | Individual patient record | Standard Opacus per-sample gradient clipping/noising during local training |
| B. True client-level DP | Entire ambulance/client | Custom mechanism: aggregate local gradients into one client update, clip the whole update to norm C, add Gaussian noise at the client boundary, account across communication rounds (McMahan et al., 2018) |

**Chosen: Option A, record-level DP.** Matches the supervisor's stated
preference (easier to implement, well-documented, defensible at Master's
level) and standard Opacus tooling named as M3's baseline code, without
needing a custom cross-round client-level accountant.

### 1.1 What one DP record is
**One DP record = one patient**, not one 30-second window. This distinction
matters because a naive per-window definition would undercount real privacy
loss: each patient contributes many windows (up to `max_windows_per_patient`
in `m3_data.py`), and an adversary observing the trained model gets signal
about that patient from all of them, not just one.

**Mechanism to make Opacus' standard per-example accounting valid at the
patient level, not the window level:** within each local training round, at
most **one window per patient** is sampled into that round's training batch
(uniformly at random among that patient's available windows, rotating
selection round to round). Under this constraint, each "example" Opacus
clips and noises during a round corresponds to exactly one patient, so its
standard per-example RDP/PRV accounting is directly interpretable as
per-patient (record-level) DP — no custom group-privacy composition is
needed. Multiple windows from the same patient across *different* rounds are
handled by the accountant's normal cross-step composition, the same way it
already handles any record appearing in multiple SGD steps.

### 1.2 Where clipping and noise are applied
Local, per client, during each ambulance's own training rounds — standard
Opacus DP-SGD: per-patient gradient computed (per §1.1's one-window-per-
patient sampling), clipped to L2 norm C, Gaussian noise with multiplier σ
added to the batch-summed clipped gradients, before the local optimizer step.
Model updates leaving a client (to the FLARE aggregator) are therefore
already the output of a DP mechanism — no additional noise is added
centrally (local DP, unchanged from v1 §2).

## 2. Local vs. Central DP
Unchanged from v1: **local**. Each client runs local DP-SGD before sending
updates to the aggregator; matches the roadmap's operational-layer framing
and means raw physiological data never needs to leave the originating client
even transiently for privacy purposes.

## 3. Secure Aggregation
Unchanged from v1: **not assumed**. Local DP already gives each client's
update a standalone per-patient privacy guarantee independent of what the
aggregator does with it (§1.1), so the core result does not depend on
trusting the aggregator. Secure aggregation remains a valid optional
strengthening, not assumed for the required comparison.

## 4. Reported (epsilon, delta) — Record-level or Client-level
**Record-level**, following directly from §1: the guarantee protects each
individual patient's inclusion in training, at ε = 1.0 (fixed by the
roadmap) and δ derived below.

## 5. Simulated Ambulance Clients
Unchanged from v1: **K = 10** simulated ambulance clients, each holding a
disjoint patient subset (a patient's data lives on exactly one client).
Sized to the currently available 81 complete patients (~8/client); revisit
once the full 198-patient download finishes. K itself is not the basis for
δ below (see §6.1) — that was v1's mistake.

## 6. Required DP Parameters

### 6.1 δ — revised and justified
**Rejected in v1:** δ = 0.05, justified only as "< 1/K" for K=10 clients.
The supervisor's objection is correct: this ties δ to the *number of
clients*, but the privacy unit is now the *patient* (§1), and 1/K ≈ 0.1 is
nowhere near "cryptographically small" — a δ that large means a real,
non-negligible chance of a complete privacy failure for some patient.

**Revised derivation:** standard DP convention is δ should be small relative
to the number of protected records N, typically δ ≪ 1/N (Dwork & Roth).
Here N is the total number of *patients* whose data will ultimately be used
across the federated system (not per-client, since the guarantee in §4 is
per-patient over the whole training process, and each patient's data lives
on exactly one client by construction in §5) — using the roadmap's full
target cohort, **N = 198**, gives 1/N ≈ 0.00505. Choosing δ meaningfully
below that (roughly an order of magnitude) and rounding to a conventional
value:

**Proposed: δ = 1e-4.**

This will be re-derived if N changes materially (e.g. if the final usable
cohort after quality filtering ends up well below 198) and re-validated
empirically by the pilot in §6.3, not treated as final until then.

### 6.2 Mechanism and accounting method (summary)
- **Privacy unit:** patient (record-level), per §1.1.
- **Sampling:** at most one window per patient per local training round,
  uniformly sampled among that patient's windows.
- **Clipping/noising:** Opacus DP-SGD, L2 clipping norm C (to be set from a
  pilot — see §6.3), Gaussian mechanism with noise multiplier σ.
- **Composition:** Opacus' own accountant (RDP or PRV accountant, whichever
  Opacus' installed version defaults to) composes privacy loss across all
  local SGD steps within a client, across all federated rounds that client
  participates in.
- **Threat model defended against:** an honest-but-curious FLARE aggregator,
  or an eavesdropper on the client-to-aggregator communication channel,
  attempting membership inference (was a given patient's data used in
  training this client's local model?) or reconstruction of a given
  patient's physiological windows from observed model updates.

### 6.3 Privacy–utility pilot (required before the full DP-federated run)
Per the supervisor's decision, before Step 3 (DP-Federated LSTM-AE) can run
at full scale:
1. Run on a **2–3 client subset** of the K=10 simulated ambulances.
2. Set clipping norm C from an unclipped per-patient gradient-norm
   percentile observed on that subset (not guessed).
3. With ε = 1.0, δ = 1e-4 (§6.1), and a chosen round count for the pilot,
   let Opacus' accountant derive the required noise multiplier σ.
4. Verify the accountant reports a valid (ε, δ) for the planned round count.
5. Verify the model still shows a decreasing training loss under that noise
   level (i.e., the signal isn't drowned).
6. If the noise drowns the signal: report it explicitly and propose a
   mitigation — increase K (more, smaller clients diluting each patient's
   individual signal contribution differently), reconsider Option A vs. B,
   or (only with the supervisor's sign-off, since the roadmap fixes ε = 1.0)
   revisit ε.

### 6.4 Parameters still open (mechanically dependent on the pilot, not guessed)
| Parameter | Value | Status |
|---|---|---|
| Privacy budget ε | 1.0 | Fixed by roadmap |
| Failure probability δ | 1e-4 | Derived in §6.1, to be confirmed by pilot |
| Sampling rate per round | 1 window/patient/round ÷ client's patient count | Mechanism fixed (§6.2); exact numeric rate depends on final per-client patient counts once K=10 partition is drawn |
| Total federated rounds | Not yet set | To be fixed from pilot convergence behavior (§6.3) |
| Clipping norm C | Not yet set | To be set from pilot gradient-norm percentile (§6.3) |
| Noise multiplier σ | Not yet set | Derived by Opacus' accountant from the above, not an independent choice |

## 7. Expected privacy–utility trade-off
Record-level DP-SGD at ε=1.0 adds per-patient gradient clipping and Gaussian
noise on top of the already-small per-client local datasets (~8 patients at
the current 81-patient stage). Expect the federated-with-DP regime to show
measurably higher reconstruction error / lower anomaly-detection utility
than both the centralized and non-DP-federated regimes, especially at the
current small per-client N — this gap, not a specific number, is the
quantity the pilot and full comparison are meant to characterize, per the
roadmap's own utility-vs-privacy framing for M3.

---
**Status**: Revised per supervisor decision (17 Aug 2026); resubmitted for
approval. Section 6.3 (pilot) has not been run yet — this document proposes
its design, execution is the next step once Sections 1–2 and 6.1 are
approved.
