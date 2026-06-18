import subprocess
import os


def run_lammps(dump_path, datafile_path, eps_pp=0.3, kappa=0.0):
    """
    Run a single-chain (no solvent) LAMMPS MD simulation with a Langevin thermostat.

    The production run length (prun) is defined in simulation.lammps.

    Parameters:
    - dump_path: str, output folder for the log, trajectory, and final state
    - datafile_path: str, path to the LAMMPS input data file
    - eps_pp: float, polymer-polymer LJ interaction strength
    - kappa: float, cosine-angle bending stiffness (0.0 = fully flexible chain)

    Returns:
    - dict: paths to {"dump_files", "final_config", "log"}
    """
    os.makedirs(dump_path, exist_ok=True)

    script_path = os.path.join(os.path.dirname(__file__), "simulation.lammps")
    lammps_bin = os.environ.get("LAMMPS_BIN", "lmp_serial")
    cmd = [
        lammps_bin,
        "-in", script_path,
        "-var", "dump_path", dump_path,
        "-var", "datafile_path", datafile_path,
        "-var", "eps_pp", str(eps_pp),
        "-var", "kappa", str(kappa),
    ]

    print(f"Running LAMMPS: {' '.join(cmd)}")
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True,
        bufsize=1,
    )

    # Stream LAMMPS output in real time
    assert process.stdout is not None
    for line in process.stdout:
        print(line, end="", flush=True)
    process.wait()

    if process.returncode != 0:
        raise subprocess.CalledProcessError(process.returncode, cmd)

    return {
        "dump_files": f"{dump_path}/coord/dump.*.txt",
        "final_config": f"{dump_path}/final_state.data",
        "log": f"{dump_path}/log.lammps",
    }
