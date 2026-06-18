"""Run a batch of rand-mode simulations locally, in parallel.

Single-chain runs are fast and each LAMMPS process is serial, so we run several
at once across local CPU cores instead of submitting to SLURM. Edit the
hard-coded settings below.

    python run_local_batch.py
"""
import concurrent.futures
import os

from run_simulation import run_simulation, DEFAULT_OUTPUT_DIR

# ---- settings ----
INIT_RUN = 701          # first run index (inclusive)
FINAL_RUN = 999         # last run index (inclusive)
CHAIN_LENGTH = 100
BOX_SIZE = 40.0
OUTPUT_DIR = DEFAULT_OUTPUT_DIR
# Number of simulations to run concurrently. Each LAMMPS process is serial, so a
# good default is roughly the number of available cores. None -> os.cpu_count().
NUM_WORKERS = None


def _run_one(run):
    """Worker: run a single simulation and return its result (or raise)."""
    return run_simulation(
        mode="rand",
        chain_length=CHAIN_LENGTH,
        box_size=BOX_SIZE,
        run=run,
        output_dir=OUTPUT_DIR,
    )


if __name__ == "__main__":
    runs = list(range(INIT_RUN, FINAL_RUN + 1))
    workers = NUM_WORKERS or os.cpu_count() or 1
    print(f"Running {len(runs)} simulations (runs {INIT_RUN}-{FINAL_RUN}) "
          f"with {workers} parallel workers")

    failures = []
    completed = 0
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
        future_to_run = {executor.submit(_run_one, run): run for run in runs}
        for future in concurrent.futures.as_completed(future_to_run):
            run = future_to_run[future]
            completed += 1
            try:
                future.result()
                print(f"[{completed}/{len(runs)}] run {run} done")
            except Exception as exc:  # noqa: BLE001 - keep the batch going
                failures.append(run)
                print(f"[{completed}/{len(runs)}] run {run} FAILED: {exc}")

    if failures:
        print(f"\nFinished with {len(failures)} failure(s): {sorted(failures)}")
    else:
        print(f"\nAll {len(runs)} runs completed successfully")
