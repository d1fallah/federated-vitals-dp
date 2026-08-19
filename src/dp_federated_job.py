"""
M3 Step 3a — DP pilot job (2-3 client pilot, approved 18 Aug 2026)
======================================================================
Adapted from NVIDIA FLARE's own hello-dp job.py: same FedAvgRecipe pattern
with aggregator_data_kind=DataKind.WEIGHT_DIFF (matching the client's
ParamsType.DIFF weight-diff reporting). NOT the full K=10 DP-federated run
-- n_clients is capped to the pilot scope authorized by the supervisor.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lstm_autoencoder import LSTMAutoencoder

from nvflare.apis.dxo import DataKind
from nvflare.app_opt.pt.recipes.fedavg import FedAvgRecipe
from nvflare.recipe import SimEnv


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_clients", type=int, default=3, help="Pilot scope: 2 or 3 only, per supervisor approval")
    parser.add_argument("--num_rounds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--target_epsilon", type=float, default=1.0)
    parser.add_argument("--target_delta", type=float, default=1e-4)
    args = parser.parse_args()

    if args.n_clients > 3:
        raise ValueError(
            f"n_clients={args.n_clients} exceeds the supervisor-approved pilot scope (2-3 clients only). "
            "The full K=10 DP-federated run is held until the pilot checklist is reported and approved."
        )

    project_root = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
    os.environ["M3_PROJECT_ROOT"] = project_root

    train_args = f"--epochs {args.epochs} --target_epsilon {args.target_epsilon} --target_delta {args.target_delta}"

    recipe = FedAvgRecipe(
        name="m3-dp-pilot",
        min_clients=args.n_clients,
        num_rounds=args.num_rounds,
        model=LSTMAutoencoder(n_channels=3, hidden_size=32, latent_size=16),
        train_script="m3_dp_pilot_client.py",
        train_args=train_args,
        aggregator_data_kind=DataKind.WEIGHT_DIFF,
    )

    env = SimEnv(num_clients=args.n_clients)
    run = recipe.execute(env)
    print()
    print("Job Status is:", run.get_status())
    print("Result can be found in:", run.get_result())


if __name__ == "__main__":
    main()
