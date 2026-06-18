#!/bin/bash
#SBATCH -A MAT226
#SBATCH -N 32
#SBATCH -t 36:00:00

# Task-farm single-chain polymer MD simulations (precision mode) over an explicit
# grid of epsilon and kappa values, one independent simulation per core. Each run
# folder is named by its parameters and contains parameters.json.
#
# Usage:
#   sbatch run_precision.sbatch

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

# ---- parameter grid (edit as needed) ----
chain_length=100
box_size=40
eps_list="0.1 0.3 0.5 1.0"
kappa_list="0 1 2 5 10"

for eps in ${eps_list}
do
    for kappa in ${kappa_list}
    do
        echo "running precision L:${chain_length} box:${box_size} eps:${eps} kappa:${kappa} folder:${folder}"
        srun -n1 -N1 -c1 --mem-per-cpu=8G --exclusive \
            python3 ${driver} --mode precision \
                --chain_length ${chain_length} --box_size ${box_size} \
                --eps ${eps} --kappa ${kappa} --output_dir ${folder} &
    done
done
wait
