"""
M3 Step 3 — DP-federated job, K-sweep (K=10, 5, 3), per supervisor guidance 19 Aug 2026
===========================================================================================
Adapted from NVIDIA FLARE's own hello-dp job.py: same FedAvgRecipe pattern
with aggregator_data_kind=DataKind.WEIGHT_DIFF (matching the client's
ParamsType.DIFF weight-diff reporting).

The 18 Aug decision capped this at a 2-3 client PILOT (a subset of an
eventual K=10 partition) to verify the DP mechanism cheaply. The 19 Aug
"Research vs. Deployment" guidance explicitly directs running a K-sweep
(K=10, 5, 3) as the research-track diagnostic for the batch-size/utility
problem -- for each K, ALL K clients participate (not a 2-3 subset), since
K itself is now the experimental variable being measured. --k_clients
controls both the ambulance-partition size (passed to the client via
M3_K_CLIENTS) and defaults n_clients to match, i.e. every client in that
partition runs.
"""
import argparse
import os
import sys

# Must be set before `import torch` anywhere in this process (NVFlare's SimEnv
# runs every client in-thread, sharing this same process/environment) --
# PyTorch caches CUDA device visibility at first import. Setting this makes
# CUDA genuinely invisible at the OS/driver level, rather than relying on
# device="cpu" call sites alone: this session hit reproducible CUDA
# out-of-memory / "device busy" errors specifically on the DP-federated path
# even with every device="cpu", most likely from Opacus/DPLSTM setup code
# eagerly touching CUDA when it's merely available, not necessarily used.
# "-1" is the sentinel that actually hides the GPU here -- verified directly:
# CUDA_VISIBLE_DEVICES="" left torch.cuda.is_available() True on this
# PyTorch/driver combination, "-1" correctly makes it False.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lstm_autoencoder import LSTMAutoencoder

from nvflare.apis.dxo import DataKind
from nvflare.app_opt.pt.recipes.fedavg import FedAvgRecipe
from nvflare.recipe import SimEnv


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--k_clients", type=int, default=10, help="Ambulance-partition size K (10, 5, or 3 for the sweep)")
    parser.add_argument("--n_clients", type=int, default=None, help="How many of the K clients to actually run; defaults to K (all of them)")
    parser.add_argument("--num_rounds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--target_epsilon", type=float, default=1.0)
    parser.add_argument("--target_delta", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42,
                         help="Patient split / client assignment seed. Default 42 reproduces the original recorded results exactly; other values (e.g. 123, 7) are for multi-seed confidence-interval reporting.")
    args = parser.parse_args()
    n_clients = args.n_clients if args.n_clients is not None else args.k_clients

    project_root = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
    os.environ["M3_PROJECT_ROOT"] = project_root
    os.environ["M3_K_CLIENTS"] = str(args.k_clients)
    os.environ["M3_SEED"] = str(args.seed)

    train_args = f"--epochs {args.epochs} --target_epsilon {args.target_epsilon} --target_delta {args.target_delta}"

    seed_suffix = "" if args.seed == 42 else f"-seed{args.seed}"
    recipe = FedAvgRecipe(
        name=f"m3-dp-federated-k{args.k_clients}{seed_suffix}",
        min_clients=n_clients,
        num_rounds=args.num_rounds,
        model=LSTMAutoencoder(n_channels=3, hidden_size=32, latent_size=16),
        train_script="dp_federated_client.py",
        train_args=train_args,
        aggregator_data_kind=DataKind.WEIGHT_DIFF,
    )

    env = SimEnv(num_clients=n_clients)
    run = recipe.execute(env)
    print()
    print("Job Status is:", run.get_status())
    print("Result can be found in:", run.get_result())


if __name__ == "__main__":
    main()
