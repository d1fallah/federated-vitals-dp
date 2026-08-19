"""
M3 Step 3a — DP pilot client-side local training
====================================================
2-3 client pilot only, per the supervisor's decision of 18 Aug 2026 -- NOT
the full K=10 DP-federated run, which stays held until this pilot's
checklist (M3_Supervisor_Decision. 2.docx Section 4) is reported and
approved.

Adapted directly from NVIDIA FLARE's own hello-dp example
(module 3/REPOS BASELINE/NVFlare-main/NVFlare-main/examples/hello-world/hello-dp/client.py):
same PrivacyEngine-created-once-at-round-0 + make_private_with_epsilon +
"_module." state_dict prefix + weight-diff pattern for combining Opacus
with FLARE's FedAvg. Three things added on top, because the checklist asks
for verified mechanism, not just described intent:

  1. One-window-per-patient-per-round sampling, asserted and logged every
     round (checklist item 1: "no client batch in any round contains two
     windows from the same patient -- every round").
  2. A pre-registered clipping norm C, derived from a single unclipped,
     unnoised dry backward pass on round-0 data, fixed BEFORE any DP
     training starts and never adjusted afterward (checklist item 4).
  3. Real achieved (epsilon, delta) reported every round via the SAME
     PrivacyEngine/accountant object across all rounds (checklist item 2:
     cross-round composition, not a per-round reset), plus training loss
     and the Steps 1/2 synthetic-anomaly evaluation protocol on the
     resulting model (checklist item 3), done in evaluate_dp_pilot_model.py.

Standard nn.LSTM is not compatible with Opacus's per-sample-gradient hooks
(confirmed via opacus.validators.ModuleValidator -- it explicitly rejects
nn.LSTM and requires opacus.layers.DPLSTM). ModuleValidator.fix() is used to
convert the model at runtime; DPLSTM is a verified drop-in replacement
(identical state_dict keys/shapes, ~1e-7 numerical difference from an
nn.LSTM with the same weights).
"""
import argparse
import json
import os
import re
import sys

import numpy as np
import torch
from opacus import PrivacyEngine
from opacus.grad_sample import GradSampleModule
from opacus.validators import ModuleValidator

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data import list_complete_patients, build_dataset, assign_ambulance_clients
from lstm_autoencoder import LSTMAutoencoder

import nvflare.client as flare
from nvflare.app_common.abstract.fl_model import ParamsType

K_CLIENTS = 10  # matches M3_DP_Threat_Model_Gate.md Section 5 -- unchanged even
                # though only 2-3 of these 10 clients run in the pilot, so pilot
                # client_ids map to the same patient partitions the eventual
                # full K=10 run will use.
SEED = 42
WINDOW_SEC = 10
CLIP_PERCENTILE = 95.0  # pre-registered before any DP training; must not change post-hoc


class SwappableWindowDataset(torch.utils.data.Dataset):
    """Holds exactly one window per patient for the current round. Content is
    swapped in place between rounds (set_windows) so the SAME wrapped
    DataLoader / PrivacyEngine from round 0 stays valid for every later
    round -- length (and therefore Opacus' sample_rate) never changes,
    only which specific windows are in it."""

    def __init__(self, windows):
        self.windows = torch.as_tensor(windows, dtype=torch.float32)

    def set_windows(self, windows):
        self.windows = torch.as_tensor(windows, dtype=torch.float32)

    def __len__(self):
        return self.windows.shape[0]

    def __getitem__(self, idx):
        w = self.windows[idx]
        return w, w  # autoencoder: input == target


def sample_one_window_per_patient(patient_windows, rng):
    """patient_windows: {patient_id: np.ndarray [n_windows_i, window_len, n_channels]}.
    Returns (windows_array, patient_ids_used), exactly one window per patient."""
    chosen, ids_used = [], []
    for pid, windows in patient_windows.items():
        idx = int(rng.integers(0, windows.shape[0]))
        chosen.append(windows[idx])
        ids_used.append(pid)
    return np.stack(chosen), ids_used


def derive_clip_norm(reference_model, windows, percentile, device="cpu"):
    """Pre-registered clipping norm: `percentile` of per-example L2 gradient
    norms from ONE unclipped, unnoised dry backward pass, computed on a
    throwaway copy of the model so the real training model is untouched."""
    temp_model = ModuleValidator.fix(
        LSTMAutoencoder(n_channels=windows.shape[-1], hidden_size=32, latent_size=16)
    )
    temp_model.load_state_dict(reference_model.state_dict())
    gsm = GradSampleModule(temp_model)
    gsm.to(device)
    X = torch.as_tensor(windows, dtype=torch.float32, device=device)
    gsm.zero_grad()
    recon = gsm(X)
    loss = torch.nn.functional.mse_loss(recon, X)
    loss.backward()
    sq_norms = None
    for p in gsm.parameters():
        gs = getattr(p, "grad_sample", None)
        if gs is None:
            continue
        gs = gs.reshape(gs.shape[0], -1)
        sq = (gs ** 2).sum(dim=1)
        sq_norms = sq if sq_norms is None else sq_norms + sq
    per_example_norms = sq_norms.sqrt().detach().cpu().numpy()
    C = float(np.percentile(per_example_norms, percentile))
    return C, per_example_norms


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=4)  # local epochs per round; 4*5 rounds=20
                                                            # epochs_param, matching the empirically
                                                            # validated subsampling config (sigma~5.7 at n=8)
    parser.add_argument("--target_epsilon", type=float, default=1.0)
    parser.add_argument("--target_delta", type=float, default=1e-4)
    args = parser.parse_args()
    PROJECT_ROOT = os.environ["M3_PROJECT_ROOT"]

    flare.init()
    sys_info = flare.system_info()
    site_name = sys_info["site_name"]
    m = re.search(r"(\d+)$", site_name)
    client_id = (int(m.group(1)) - 1) % K_CLIENTS if m else 0
    rng = np.random.default_rng(SEED + client_id)

    complete = list_complete_patients(PROJECT_ROOT)
    clients = assign_ambulance_clients(complete, k=K_CLIENTS, seed=SEED)
    my_patients = clients[client_id]
    print(f"site={site_name} -> ambulance client_id={client_id} -> {len(my_patients)} patients")

    patient_windows = {}
    for pid in my_patients:
        X_p, _ = build_dataset(PROJECT_ROOT, [pid], window_sec=WINDOW_SEC, max_windows_per_patient=80)
        if X_p.shape[0] > 0:
            patient_windows[pid] = X_p
    if len(patient_windows) < 2:
        raise RuntimeError(
            f"site={site_name}: only {len(patient_windows)} usable patients, need >=2 for the pilot"
        )
    print(f"site={site_name}: {len(patient_windows)} usable patients, "
          f"pool sizes {[w.shape[0] for w in patient_windows.values()]}")

    device = torch.device("cpu")
    model = ModuleValidator.fix(LSTMAutoencoder(n_channels=3, hidden_size=32, latent_size=16))

    privacy_engine = None
    optimizer = None
    dataset_obj = None
    train_loader = None
    clip_norm = None
    round_log = []

    while flare.is_running():
        input_model = flare.receive()
        round_idx = input_model.current_round
        total_rounds = input_model.total_rounds
        print(f"site={site_name}, current_round={round_idx}/{total_rounds}")

        windows, ids_used = sample_one_window_per_patient(patient_windows, rng)
        ok = len(ids_used) == len(set(ids_used))
        round_log.append({"round": round_idx, "n_windows": len(ids_used), "n_unique_patients": len(set(ids_used)), "ok": ok})
        print(f"site={site_name}: round {round_idx} batch = {len(ids_used)} windows, "
              f"{len(set(ids_used))} unique patients -> {'OK' if ok else 'VIOLATION'}")
        assert ok, f"site={site_name}: duplicate patient in round {round_idx}'s batch -- one-window-per-patient constraint violated"

        if round_idx == 0:
            # Load the actual received global round-0 weights BEFORE deriving
            # the clip norm, so the dry pass reflects the real starting
            # model, not this client's independently-random local init
            # (model is still the plain, unwrapped module at this point).
            # strict=False: the server's initial model is plain nn.LSTM-based
            # (job.py never runs it through ModuleValidator.fix()), so its
            # checkpoint only has the standard weight_ih_l0-style keys. Our
            # client model is DPLSTM-based, which also exposes internal
            # encoder.l0.ih.weight-style keys alongside the same standard
            # ones -- verified these are aliased/derived, not independent
            # parameters (loading only the standard keys reproduces the
            # source model's output to ~3e-8, pure float rounding).
            missing, unexpected = model.load_state_dict(input_model.params, strict=False)
            assert not unexpected, f"unexpected keys when loading global round-0 weights: {unexpected}"

            clip_norm, dry_norms = derive_clip_norm(model, windows, CLIP_PERCENTILE, device)
            print(f"site={site_name}: pre-registered clipping norm C = {clip_norm:.5f} "
                  f"({CLIP_PERCENTILE}th percentile of {len(dry_norms)} unclipped per-patient "
                  f"gradient norms, dry pass; fixed now, will not be tuned post-hoc)")

            dataset_obj = SwappableWindowDataset(windows)
            # batch_size=1 (was len(dataset_obj), i.e. the whole per-round
            # pool = q=1.0, no subsampling at all). Verified empirically this
            # was the dominant cause of the huge noise multiplier: with
            # Opacus's Poisson-sampling DataLoader, q=1.0 gets ZERO privacy
            # amplification from subsampling -- one of the worst possible
            # DP-SGD configurations. batch_size=1 gives q=1/n_patients per
            # step (real subsampling), cutting the accountant-derived sigma
            # roughly in half at this dataset scale (measured: 11.2 -> 5.7 at
            # n=8, target eps=1.0/delta=1e-4). The one-window-per-patient
            # invariant still holds automatically: this dataset already has
            # at most one window per patient by construction (see
            # sample_one_window_per_patient), so ANY subset Opacus draws
            # from it also has at most one window per patient.
            train_loader = torch.utils.data.DataLoader(dataset_obj, batch_size=1, shuffle=False)
            raw_optimizer = torch.optim.Adam(model.parameters(), lr=3e-3)
            # epochs param to make_private_with_epsilon = passes over the
            # dataloader, NOT total optimizer steps directly (that distinction
            # only vanished before because batch_size==dataset_size made
            # len(dataloader)==1). Actual total steps ~= this * n_patients.
            epochs_param = args.epochs * total_rounds

            privacy_engine = PrivacyEngine()
            model, optimizer, train_loader = privacy_engine.make_private_with_epsilon(
                module=model,
                optimizer=raw_optimizer,
                data_loader=train_loader,
                epochs=epochs_param,
                target_epsilon=args.target_epsilon,
                target_delta=args.target_delta,
                max_grad_norm=clip_norm,
            )
            # model/optimizer are now Opacus-wrapped, but wrapping preserves
            # the parameter values already loaded above -- no re-load needed.
            print(f"site={site_name}: noise multiplier sigma = {optimizer.noise_multiplier:.5f} "
                  f"(Opacus-derived for target eps={args.target_epsilon}, delta={args.target_delta}, "
                  f"epochs_param={epochs_param} over {len(train_loader)} steps/epoch "
                  f"~= {epochs_param * len(train_loader)} total steps)")
        else:
            dataset_obj.set_windows(windows)
            global_params = {f"_module.{k}": v for k, v in input_model.params.items()}
            missing, unexpected = model.load_state_dict(global_params, strict=False)
            assert not unexpected, f"unexpected keys when loading round {round_idx} global weights: {unexpected}"

        model.to(device)
        model.train()

        loss_fn = torch.nn.MSELoss()
        epoch_losses = []
        n_empty_batches = 0
        for _ in range(args.epochs):
            for batch, target in train_loader:
                # Poisson sampling at batch_size=1 on a tiny (7-9 item)
                # dataset regularly draws empty batches (measured: ~25-40%
                # of draws per epoch at this scale). MSELoss on an empty
                # batch is 0/0 = NaN, and one NaN optimizer.step()
                # permanently corrupts the model for the rest of training --
                # this was the actual cause of every round showing
                # local_loss=nan in the first run of this subsampling
                # config. Skip empty batches rather than stepping on them.
                if batch.shape[0] == 0:
                    n_empty_batches += 1
                    continue
                batch, target = batch.to(device), target.to(device)
                optimizer.zero_grad()
                recon = model(batch)
                loss = loss_fn(recon, target)
                loss.backward()
                optimizer.step()
                epoch_losses.append(loss.item())
        if n_empty_batches:
            print(f"site={site_name}: round {round_idx} skipped {n_empty_batches} empty Poisson-sampled batches")
        avg_loss = float(np.mean(epoch_losses)) if epoch_losses else float("nan")
        print(f"site={site_name}: round {round_idx} local loss = {avg_loss:.5f}")

        achieved_epsilon = privacy_engine.get_epsilon(args.target_delta)
        print(f"site={site_name}: cumulative achieved epsilon after round {round_idx} = "
              f"{achieved_epsilon:.5f} (delta={args.target_delta})")

        # Only report back the keys the server's plain (non-DPLSTM) model
        # actually has -- the DPLSTM-fixed local model also carries derived
        # "encoder.l0.ih.weight"-style aliases (see the strict=False load
        # above) that aren't independent parameters and the server doesn't
        # know about.
        clean_params = {k.replace("_module.", ""): v for k, v in model.cpu().state_dict().items()}
        clean_params = {k: v for k, v in clean_params.items() if k in input_model.params}
        model_diff = {
            name: local_param - torch.as_tensor(input_model.params[name], dtype=local_param.dtype)
            for name, local_param in clean_params.items()
        }

        output_model = flare.FLModel(
            params=model_diff,
            params_type=ParamsType.DIFF,
            metrics={"train_loss": avg_loss, "achieved_epsilon": achieved_epsilon},
            meta={"NUM_STEPS_CURRENT_ROUND": args.epochs},
        )
        flare.send(output_model)

        if round_idx == total_rounds - 1:
            log_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"dp_pilot_round_log_{site_name}.json")
            with open(log_path, "w") as f:
                json.dump({
                    "site": site_name,
                    "client_id": client_id,
                    "n_patients": len(patient_windows),
                    "clip_norm_C": clip_norm,
                    "clip_percentile": CLIP_PERCENTILE,
                    "noise_multiplier": optimizer.noise_multiplier,
                    "target_epsilon": args.target_epsilon,
                    "target_delta": args.target_delta,
                    "final_achieved_epsilon": achieved_epsilon,
                    "round_log": round_log,
                }, f, indent=2)
            print(f"site={site_name}: wrote verification log to {log_path}")


if __name__ == "__main__":
    main()
