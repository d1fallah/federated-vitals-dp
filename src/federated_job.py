"""
M3 Step 2 — Non-DP Federated LSTM-AE job (approved 17 Aug 2026)
==================================================================
Adapted from NVIDIA FLARE's own hello-pt job.py
(module 3/REPOS BASELINE/NVFlare-main/NVFlare-main/examples/hello-world/hello-pt/job.py):
same FedAvgRecipe + SimEnv pattern, no custom server code (FedAvg is
provided by NVFlare's ScatterAndGather controller). No differential privacy
here — Step 3 (DP-Federated) is held pending the DP threat-model gate's
approval and pilot (see M3_DP_Threat_Model_Gate.md).

--network_profile {good,moderate,poor} adds the connectivity-robustness
ablation (roadmap M3 required axis 3/3, alongside privacy and utility):
a SimPy-based network_sim.py schedule is pre-generated for this run and
consulted by federated_client.py each round. Deliberately not wired into
dp_federated_job.py -- see network_sim.py and federated_client.py for why.
"""
import argparse
import os
import sys

# See dp_federated_job.py for why: must be set before `import torch` anywhere
# in this process, and "-1" (not "") is the sentinel that actually hides the
# GPU on this PyTorch/driver combination, verified directly.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lstm_autoencoder import LSTMAutoencoder
from network_sim import generate_schedule, write_schedule

from nvflare.app_opt.pt.recipes.fedavg import FedAvgRecipe
from nvflare.recipe import SimEnv


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_clients", type=int, default=10)
    parser.add_argument("--num_rounds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--network_profile", type=str, default="none", choices=["none", "good", "moderate", "poor"],
                         help="Connectivity-robustness ablation: simulate Good/Moderate/Poor network conditions (roadmap-fixed params, see network_sim.py). Default 'none' reproduces the original no-degradation baseline exactly.")
    parser.add_argument("--seed", type=int, default=42,
                         help="Patient split / client assignment / network-schedule seed. Default 42 reproduces the original recorded results exactly; other values (e.g. 123, 7) are for multi-seed confidence-interval reporting.")
    args = parser.parse_args()

    # See federated_client.py for why this is an env var, not a train_args
    # CLI argument: train_args is split with plain whitespace splitting
    # (no quote awareness), which breaks a Windows path containing spaces.
    # The simulator's clients run in-thread in this same process, so this
    # environment variable is directly visible to them.
    project_root = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
    os.environ["M3_PROJECT_ROOT"] = project_root
    os.environ["M3_SEED"] = str(args.seed)
    train_args = f"--epochs {args.epochs} --batch_size {args.batch_size} --lr {args.lr}"

    job_seed_suffix = "" if args.seed == 42 else f"-seed{args.seed}"    # FLARE job/workspace names: hyphen, matches m3-dp-federated-k{K} convention
    file_seed_suffix = "" if args.seed == 42 else f"_seed{args.seed}"  # result/log filenames: underscore, matches network_round_log_ convention
    job_name = f"m3-non-dp-federated{job_seed_suffix}"
    if args.network_profile != "none":
        job_name = f"m3-non-dp-federated-{args.network_profile}{job_seed_suffix}"
        results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
        os.makedirs(results_dir, exist_ok=True)
        schedule = generate_schedule(args.network_profile, list(range(args.n_clients)), args.num_rounds, seed=args.seed)
        schedule_path = os.path.join(results_dir, f"network_schedule_{args.network_profile}{file_seed_suffix}.json")
        write_schedule(schedule, schedule_path)
        os.environ["M3_NETWORK_PROFILE"] = args.network_profile
        os.environ["M3_NETWORK_SCHEDULE_PATH"] = os.path.abspath(schedule_path)
        print(f"Connectivity-robustness ablation active: profile={args.network_profile}, seed={args.seed}, schedule written to {schedule_path}")

    recipe = FedAvgRecipe(
        name=job_name,
        min_clients=args.n_clients,
        num_rounds=args.num_rounds,
        model=LSTMAutoencoder(n_channels=3, hidden_size=32, latent_size=16),
        train_script="federated_client.py",
        train_args=train_args,
    )

    env = SimEnv(num_clients=args.n_clients)
    run = recipe.execute(env)
    print()
    print("Job Status is:", run.get_status())
    print("Result can be found in:", run.get_result())


if __name__ == "__main__":
    main()
