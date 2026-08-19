"""
M3 Step 2 — Non-DP Federated LSTM-AE job (approved 17 Aug 2026)
==================================================================
Adapted from NVIDIA FLARE's own hello-pt job.py
(module 3/REPOS BASELINE/NVFlare-main/NVFlare-main/examples/hello-world/hello-pt/job.py):
same FedAvgRecipe + SimEnv pattern, no custom server code (FedAvg is
provided by NVFlare's ScatterAndGather controller). No differential privacy
here — Step 3 (DP-Federated) is held pending the DP threat-model gate's
approval and pilot (see M3_DP_Threat_Model_Gate.md).
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lstm_autoencoder import LSTMAutoencoder

from nvflare.app_opt.pt.recipes.fedavg import FedAvgRecipe
from nvflare.recipe import SimEnv


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_clients", type=int, default=10)
    parser.add_argument("--num_rounds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-3)
    args = parser.parse_args()

    # See m3_flare_client.py for why this is an env var, not a train_args
    # CLI argument: train_args is split with plain whitespace splitting
    # (no quote awareness), which breaks a Windows path containing spaces.
    # The simulator's clients run in-thread in this same process, so this
    # environment variable is directly visible to them.
    project_root = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
    os.environ["M3_PROJECT_ROOT"] = project_root
    train_args = f"--epochs {args.epochs} --batch_size {args.batch_size} --lr {args.lr}"

    recipe = FedAvgRecipe(
        name="m3-non-dp-federated",
        min_clients=args.n_clients,
        num_rounds=args.num_rounds,
        model=LSTMAutoencoder(n_channels=3, hidden_size=32, latent_size=16),
        train_script="m3_flare_client.py",
        train_args=train_args,
    )

    env = SimEnv(num_clients=args.n_clients)
    run = recipe.execute(env)
    print()
    print("Job Status is:", run.get_status())
    print("Result can be found in:", run.get_result())


if __name__ == "__main__":
    main()
