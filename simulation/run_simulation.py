"""Driver for single-chain polymer MD simulations.

Two modes:
  precision -- run with explicitly given epsilon and kappa
  rand      -- randomly sample epsilon and kappa (for generating training data),
               seeded reproducibly by the run index

chain_length and box_size are always given explicitly. Each run writes all
outputs (initial config, parameters.json, log, trajectory, final state) into a
single run folder under the output directory.

Examples:
  python run_simulation.py --mode precision --chain_length 100 --box_size 40 --eps 0.3 --kappa 2.0
  python run_simulation.py --mode rand --chain_length 100 --box_size 40 --run 0
"""
import argparse
import json
import os

import numpy as np

from generate_polymer import generate_linear_polymer_config
from run_lammps import run_lammps

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_OUTPUT_DIR = os.path.join(REPO_ROOT, "data", "scratch_local")

BASE_SEED = 67890

# rand-mode uniform sampling ranges
EPS_RANGE = (0.0, 2.0)
KAPPA_RANGE = (0.0, 10.0)


def sample_parameters(run):
    """Draw (eps_pp, kappa) for rand mode, reproducibly seeded by the run index."""
    rng = np.random.default_rng(BASE_SEED + run)
    eps_pp = float(rng.uniform(*EPS_RANGE))
    kappa = float(rng.uniform(*KAPPA_RANGE))
    return eps_pp, kappa


def run_folder_name(mode, chain_length, box_size, eps_pp, kappa, run):
    """precision encodes eps/kappa in the name; rand uses a run index (those are random)."""
    if mode == "rand":
        return f"linear_L{chain_length}_box{box_size:g}_run{run}"
    return f"linear_L{chain_length}_box{box_size:g}_eps{eps_pp:g}_kappa{kappa:g}"


def run_simulation(mode, chain_length, box_size, eps_pp=None, kappa=None, run=0, output_dir=DEFAULT_OUTPUT_DIR):
    """Generate a chain, record the parameters, and run the MD simulation."""
    if mode == "rand":
        eps_pp, kappa = sample_parameters(run)
    elif mode == "precision":
        if eps_pp is None or kappa is None:
            raise ValueError("precision mode requires both --eps and --kappa")
    else:
        raise ValueError(f"Unknown mode: {mode}")

    run_dir = os.path.join(output_dir, run_folder_name(mode, chain_length, box_size, eps_pp, kappa, run))
    os.makedirs(run_dir, exist_ok=True)

    # Record the parameters actually used, for downstream bookkeeping
    parameters = {
        "mode": mode,
        "chain_length": chain_length,
        "box_size": box_size,
        "epsilon": eps_pp,
        "kappa": kappa,
    }
    if mode == "rand":
        parameters["run"] = run
        parameters["seed"] = BASE_SEED + run
    with open(os.path.join(run_dir, "parameters.json"), "w") as f:
        json.dump(parameters, f, indent=2)
    print(f"Parameters: {parameters}")

    # Generate the initial configuration into the run folder
    datafile = generate_linear_polymer_config(chain_length, box_size=box_size, output_dir=run_dir)

    # Run the simulation, writing all outputs into the same folder
    return run_lammps(run_dir, datafile, eps_pp=eps_pp, kappa=kappa)


def parse_args():
    p = argparse.ArgumentParser(description="Run a single-chain polymer MD simulation.")
    p.add_argument("--mode", choices=["precision", "rand"], required=True)
    p.add_argument("--chain_length", type=int, required=True)
    p.add_argument("--box_size", type=float, required=True)
    p.add_argument("--eps", type=float, default=None, help="LJ epsilon (required in precision mode)")
    p.add_argument("--kappa", type=float, default=None, help="bending stiffness (required in precision mode)")
    p.add_argument("--run", type=int, default=0, help="run index / random seed offset (rand mode)")
    p.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    results = run_simulation(
        mode=args.mode,
        chain_length=args.chain_length,
        box_size=args.box_size,
        eps_pp=args.eps,
        kappa=args.kappa,
        run=args.run,
        output_dir=args.output_dir,
    )
    print(results)
