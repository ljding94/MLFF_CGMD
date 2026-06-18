import numpy as np
import os
import json

DEFAULT_BOX_SIZE = 50.0


def generate_gaussian_chain(chain_length, bond_length, angle_mean, angle_std):
    """
    Generate a Gaussian chain with specified bond length and angle distribution.

    Parameters:
    - chain_length: int, number of atoms in the chain
    - bond_length: float, distance between consecutive atoms
    - angle_mean: float, mean bond angle in radians
    - angle_std: float, standard deviation of bond angle in radians

    Returns:
    - list of np.array: positions of atoms in the chain
    """
    if chain_length < 2:
        return [np.array([0.0, 0.0, 0.0])]

    positions = [np.array([0.0, 0.0, 0.0])]  # Start at origin

    # First two atoms: along x-axis
    positions.append(np.array([bond_length, 0.0, 0.0]))

    # Generate remaining atoms
    for i in range(2, chain_length):
        # Sample bond angle from Gaussian distribution
        bond_angle = np.random.normal(angle_mean, angle_std)
        bond_angle = np.clip(bond_angle, 0.1, np.pi - 0.1)  # Keep within reasonable bounds

        # Get the previous two positions to determine the plane
        pos_prev2 = positions[i - 2]
        pos_prev1 = positions[i - 1]

        # Vector from prev2 to prev1 (previous bond)
        prev_bond = pos_prev1 - pos_prev2
        prev_bond = prev_bond / np.linalg.norm(prev_bond)

        # Generate a random torsion angle for 3D flexibility
        torsion_angle = np.random.uniform(0, 2 * np.pi)

        # Create rotation matrix to generate new bond direction
        # First, create a perpendicular vector to the previous bond
        if abs(prev_bond[0]) > 0.1:
            perp = np.array([-prev_bond[1] - prev_bond[2], prev_bond[0], prev_bond[0]])
        else:
            perp = np.array([prev_bond[1], -prev_bond[0] - prev_bond[2], prev_bond[1]])
        perp = perp / np.linalg.norm(perp)

        # Second perpendicular vector
        perp2 = np.cross(prev_bond, perp)

        # New bond direction: rotate around previous bond by bond_angle, then around torsion
        cos_angle = np.cos(bond_angle)
        sin_angle = np.sin(bond_angle)

        # Rotate previous bond by bond_angle around perp axis
        new_bond = cos_angle * prev_bond + sin_angle * np.cos(torsion_angle) * perp + sin_angle * np.sin(torsion_angle) * perp2

        new_bond = new_bond / np.linalg.norm(new_bond)  # New position
        new_pos = pos_prev1 + bond_length * new_bond
        positions.append(new_pos)

    return positions


def generate_linear_polymer_config(chain_length, box_size=DEFAULT_BOX_SIZE, output_dir=None):
    """
    Generates LAMMPS datafile for a linear polymer (Gaussian chain).

    Parameters:
    - chain_length: int, number of beads in the chain
    - box_size: float, size of simulation box
    - output_dir: str, directory to save the datafile (default: current directory)

    Returns:
    - str: path to the generated datafile
    """
    # Metadata
    metadata = {
        "type": "linear",
        "chain_length": chain_length,
        "box_size": box_size
    }

    # Initialize lists
    positions = []
    bonds = []
    angles = []
    atom_id = 1
    bond_id = 1
    angle_id = 1

    # Gaussian chain parameters
    bond_length = 1.0
    angle_mean = 0.0
    angle_std = np.pi * 60.0 / 180.0  # 60 degree standard deviation

    # Generate chain as Gaussian chain
    chain_positions = generate_gaussian_chain(chain_length, bond_length, angle_mean, angle_std)

    # Add atoms to positions
    for pos in chain_positions:
        positions.append((atom_id, 1, 1, 0, pos[0], pos[1], pos[2]))  # Type 1 for all atoms
        atom_id += 1

    # Bonds
    for i in range(chain_length - 1):
        bonds.append((bond_id, 1, i + 1, i + 2))
        bond_id += 1

    # Angles
    for i in range(chain_length - 2):
        angles.append((angle_id, 1, i + 1, i + 2, i + 3))
        angle_id += 1

    # Center the polymer at origin
    if positions:
        com = np.mean([[pos[4], pos[5], pos[6]] for pos in positions], axis=0)
        for i, pos in enumerate(positions):
            new_pos = (pos[0], pos[1], pos[2], pos[3], pos[4] - com[0], pos[5] - com[1], pos[6] - com[2])
            positions[i] = new_pos

    # Write datafile
    if output_dir is None:
        output_dir = os.getcwd()
    datafile_path = os.path.join(output_dir, "polymer_linear.data")
    with open(datafile_path, "w") as f:
        f.write(f"# Metadata: {json.dumps(metadata)}\n")
        f.write("# Linear polymer datafile\n")
        f.write(f"{len(positions)} atoms\n")
        f.write(f"{len(bonds)} bonds\n")
        f.write(f"{len(angles)} angles\n")
        f.write("1 atom types\n")
        f.write("1 bond types\n")
        f.write("1 angle types\n")
        f.write(f"{-box_size/2} {box_size/2} xlo xhi\n")
        f.write(f"{-box_size/2} {box_size/2} ylo yhi\n")
        f.write(f"{-box_size/2} {box_size/2} zlo zhi\n")
        f.write("\nMasses\n\n")
        f.write("1 1.00\n")
        f.write("\n")
        f.write("Atoms #full\n\n")
        for atom in positions:
            f.write(f"{atom[0]} {atom[1]} {atom[2]} {atom[3]} {atom[4]} {atom[5]} {atom[6]}\n")
        f.write("\n")
        f.write("Bonds\n\n")
        for bond in bonds:
            f.write(f"{bond[0]} {bond[1]} {bond[2]} {bond[3]}\n")
        f.write("\n")
        f.write("Angles\n\n")
        for angle in angles:
            f.write(f"{angle[0]} {angle[1]} {angle[2]} {angle[3]} {angle[4]}\n")
        f.write("\n")

    print(f"Generated linear polymer with {chain_length} beads at {datafile_path}")
    return datafile_path


if __name__ == "__main__":
    generate_linear_polymer_config(chain_length=50, box_size=DEFAULT_BOX_SIZE, output_dir=os.getcwd())
