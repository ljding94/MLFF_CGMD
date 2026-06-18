#!/bin/bash
#SBATCH -A MAT226
#SBATCH -N 32
#SBATCH -t 36:00:00

# Task-farm 1024 single-chain polymer MD simulations (rand mode) across the
# allocated nodes, one independent simulation per core. In rand mode the driver
# samples epsilon and kappa (seeded by the run index) and records the parameters
# in parameters.json under each run folder. chain_length and box_size are fixed.
#
# Usage:
#   sbatch run_rand.sbatch <start_run_index>
# e.g. sbatch run_rand.sbatch 0      -> runs 0..1023
#      sbatch run_rand.sbatch 1024   -> runs 1024..2047

module load Core/24.00 openmpi/4.1.2
module load fftw
# A Python with numpy must be on PATH (e.g. module load / conda). run_simulation.py needs it.
# module load python

# LAMMPS binary used by run_lammps.py (overrides the lmp_serial default).
# Each task is single-core, so a serial-capable binary is expected.
export LAMMPS_BIN=lmp_mpi

# ---- paths (edit for your account) ----
folder=/lustre/orion/mat226/scratch/dingl/MLFF_CGMD/data_pool
driver=/ccs/home/dingl/MLFF_CGMD/simulation/run_simulation.py

mkdir -p ${folder}

# ---- run parameters ----
n1=$1               # starting run index, passed as the first argument
chain_length=100
box_size=40

for run in $(seq $n1 $((n1 + 1023)))
do
    echo "running rand L:${chain_length} box:${box_size} run:${run} folder:${folder}"
    srun -n1 -N1 -c1 --mem-per-cpu=8G --exclusive \
        python3 ${driver} --mode rand \
            --chain_length ${chain_length} --box_size ${box_size} \
            --run ${run} --output_dir ${folder} &
done
wait
