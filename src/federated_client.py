"""
M3 Step 2 — FLARE client-side local training (Non-DP Federated LSTM-AE)
==========================================================================
Approved 17 Aug 2026 to proceed now (no DP, outside the DP threat-model
gate's scope). Adapted from NVIDIA FLARE's own hello-pt example
(module 3/REPOS BASELINE/NVFlare-main/NVFlare-main/examples/hello-world/hello-pt/client.py):
same flare.init()/receive()/send() client-API pattern, CIFAR10 classification
replaced with our LSTM-AE reconstruction task and per-ambulance patient data
(m3_data.assign_ambulance_clients) instead of a shared dataset.

Connectivity-robustness ablation (M3_NETWORK_PROFILE / M3_NETWORK_SCHEDULE_PATH,
set by federated_job.py when --network_profile is not "none"): each round,
this client checks a pre-registered network_sim.py delivery schedule before
touching the received global weights. If not delivered this round, it sends
back flare.FLModel(params={}) immediately and skips local training entirely
-- verified against the vendored FedAvg controller
(nvflare/app_common/workflows/fedavg.py) that an empty params dict is the
FLARE-native "no update this round" signal, excluded from aggregation
without affecting min_clients/timeout bookkeeping. Default (no env vars set)
takes zero new code paths, reproducing the original baseline exactly.
"""
import argparse
import json
import os
import re
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data import list_complete_patients, build_dataset, assign_ambulance_clients
from lstm_autoencoder import LSTMAutoencoder, train_lstm_ae, reconstruction_error
from network_sim import load_schedule

import nvflare.client as flare

# NVFlare copies this script into a per-site sandbox job folder
# (.../simulate_job/app_site-N/custom/m3_flare_client.py) before running it,
# so PROJECT_ROOT cannot be derived from __file__'s location. It also can't
# be passed via train_args: NVFlare's task_script_runner splits train_args
# with plain str.split() (no quote awareness), which mangles a Windows path
# containing spaces ("Modules Services startup"). The client task runs
# in-thread (client_api_in_process_trainer), not a subprocess, so a plain
# environment variable set by m3_flare_job.py before recipe.execute() is
# directly visible here without going through any command-line splitting.
K_CLIENTS = 10  # matches M3_DP_Threat_Model_Gate.md Section 5
WINDOW_SEC = 10
MAX_WINDOWS_PER_PATIENT = 80


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-3)
    args = parser.parse_args()
    PROJECT_ROOT = os.environ["M3_PROJECT_ROOT"]
    SEED = int(os.environ.get("M3_SEED", "42"))  # multi-seed confidence-interval reruns; see federated_job.py --seed

    # (1) initialize NVFlare client API
    flare.init()
    sys_info = flare.system_info()
    site_name = sys_info["site_name"]  # e.g. "site-3" in simulator mode
    m = re.search(r"(\d+)$", site_name)
    client_id = (int(m.group(1)) - 1) % K_CLIENTS if m else 0

    complete = list_complete_patients(PROJECT_ROOT)
    clients = assign_ambulance_clients(complete, k=K_CLIENTS, seed=SEED)
    my_patients = clients[client_id]
    print(f"site={site_name} -> ambulance client_id={client_id} -> {len(my_patients)} patients")

    X_local, _ = build_dataset(
        PROJECT_ROOT, my_patients, window_sec=WINDOW_SEC, max_windows_per_patient=MAX_WINDOWS_PER_PATIENT
    )
    print(f"site={site_name}: {X_local.shape[0]} local training windows")

    # NOT cuda: NVFlare's SimEnv runs every client in-thread, sharing one process
    # and one CUDA context. Verified this really breaks with K=10 concurrent
    # clients (torch.AcceleratorError: CUDA error: out of memory on a 6GB card,
    # then cascading CUDNN_STATUS_EXECUTION_FAILED_CUDART in other threads once
    # one context failed) -- not a hypothetical risk, an actual crash reproduced
    # this session. These models are tiny (hidden_size=32); CPU is safe and fast
    # enough. GPU is used only in train_centralized.py, which is single-threaded.
    device = torch.device("cpu")
    model = LSTMAutoencoder(n_channels=X_local.shape[-1] if X_local.shape[0] else 3, hidden_size=32, latent_size=16)

    network_profile = os.environ.get("M3_NETWORK_PROFILE", "none")
    schedule = load_schedule(os.environ["M3_NETWORK_SCHEDULE_PATH"]) if network_profile != "none" else {}
    round_log = []

    seed_suffix = "" if SEED == 42 else f"_seed{SEED}"

    def maybe_write_network_log(round_idx, total_rounds):
        if network_profile == "none" or round_idx != total_rounds - 1:
            return
        log_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"network_round_log_{network_profile}{seed_suffix}_{site_name}.json")
        with open(log_path, "w") as f:
            json.dump({
                "network_profile": network_profile,
                "seed": SEED,
                "site": site_name,
                "client_id": client_id,
                "n_local_windows": int(X_local.shape[0]),
                "round_log": round_log,
            }, f, indent=2)
        print(f"site={site_name}: wrote network round log to {log_path}")

    while flare.is_running():
        # (2) receive global model from the FLARE server
        input_model = flare.receive()
        round_idx = input_model.current_round
        total_rounds = input_model.total_rounds
        print(f"site={site_name}, current_round={round_idx}/{total_rounds}")

        if network_profile != "none":
            outcome = schedule.get(str(round_idx), {}).get(str(client_id), {"delivered": True, "reason": "ok"})
            round_log.append({"round": round_idx, "delivered": outcome["delivered"], "reason": outcome["reason"]})
            if not outcome["delivered"]:
                print(f"site={site_name}: round {round_idx} DROPPED by network sim ({network_profile}, reason={outcome['reason']}) -- skipping training")
                flare.send(flare.FLModel(params={}, metrics={"delivered": 0.0}))
                maybe_write_network_log(round_idx, total_rounds)
                continue
            print(f"site={site_name}: round {round_idx} delivered by network sim ({network_profile})")

        model.load_state_dict(input_model.params)
        model.to(device)

        if X_local.shape[0] < args.batch_size:
            print(f"site={site_name}: only {X_local.shape[0]} windows, below batch_size={args.batch_size} -- sending model back unchanged")
            output_model = flare.FLModel(
                params={k: v.cpu() for k, v in model.state_dict().items()},
                metrics={"train_loss": float("nan"), "n_local_windows": int(X_local.shape[0])},
            )
            flare.send(output_model)
            maybe_write_network_log(round_idx, total_rounds)
            continue

        history = train_lstm_ae(model, X_local, epochs=args.epochs, lr=args.lr, batch_size=args.batch_size, device=str(device))
        local_err = reconstruction_error(model, X_local, device=str(device))
        print(f"site={site_name}: local loss {history[0]:.5f} -> {history[-1]:.5f}, mean recon error {local_err.mean():.5f}")

        # (3) send locally-trained model back to the FLARE server for FedAvg
        output_model = flare.FLModel(
            params={k: v.cpu() for k, v in model.state_dict().items()},
            metrics={"train_loss": history[-1], "recon_error_mean": float(local_err.mean())},
            meta={"NUM_STEPS_CURRENT_ROUND": args.epochs * max(1, X_local.shape[0] // args.batch_size)},
        )
        flare.send(output_model)
        maybe_write_network_log(round_idx, total_rounds)


if __name__ == "__main__":
    main()
