# This file converts the alexandria datasets to hdf5 format so it's faster to load
# These hdf5 files are also more efficient since we drop unused columns
from multiprocessing import Process
from pymatgen.entries.computed_entries import ComputedStructureEntry
import numpy as np
import h5py
import json
import bz2
import pathlib
import os


ROOT_PATH = f"{pathlib.Path(__file__).parent.resolve()}"
IN_DIR = f"{ROOT_PATH}/datasets/alexandria"
OUT_DIR = f"{ROOT_PATH}/datasets/alexandria_hdf5"


def prep_data(filename, take_max_num_examples=None):
    print(f"prepping {filename}")
    with bz2.open(f"{IN_DIR}/{filename}.json.bz2", "rt", encoding="utf-8") as fh:
        data = json.load(fh)
    if take_max_num_examples is not None:
        filename = f"{filename}_take{take_max_num_examples}"
        data["entries"] = data["entries"][:take_max_num_examples]

    entries = [ComputedStructureEntry.from_dict(i) for i in data["entries"]]
    print(f"Found {len(entries)} entries for {filename}")

    # Initialize arrays
    num_entries = len(entries)
    frac_x = []
    atomic_numbers = []
    lattice = np.zeros((num_entries, 3, 3))
    idx_start = []
    num_atoms = []

    current_index = 0

    for idx, entry in enumerate(entries):
        structure = entry.structure
        num_atoms.append(len(structure.species))
        idx_start.append(current_index)
        current_index += len(structure.species)

        atomic_numbers.extend([species.Z for species in structure.species])
        lattice[idx] = structure.lattice.matrix
        frac_x.extend(structure.frac_coords.reshape(-1, 3))

    # Convert lists to numpy arrays
    atomic_numbers = np.array(atomic_numbers, dtype=int)
    frac_x = np.array(frac_x, dtype=float)
    idx_start = np.array(idx_start, dtype=int)
    num_atoms = np.array(num_atoms, dtype=int)
        

    return atomic_numbers, lattice, frac_x, idx_start, num_atoms


def save_dataset(filename, atomic_numbers, lattice, frac_x, idx_start, num_atoms):
    # Save the data to an HDF5 file
    os.makedirs(OUT_DIR, exist_ok=True)
    with h5py.File(f"{OUT_DIR}/{filename}.h5", "w") as file:
        group = file.create_group("crystals")
        group.create_dataset("frac_x", data=frac_x)
        group.create_dataset("atomic_numbers", data=atomic_numbers)
        group.create_dataset("lattice", data=lattice)
        group.create_dataset("idx_start", data=idx_start)
        group.create_dataset("num_atoms", data=num_atoms)


def rotate_lattice_about_origin(lattice: np.ndarray) -> np.ndarray:
    lower_south_west_corner = np.min(lattice, axis=1)
    lattice -= np.repeat(lower_south_west_corner, 3, axis=0).reshape(-1,3,3)
    rotation_matrix = np.array(
        [[1, 0, 0], [0, 0, -1], [0, 1, 0]]
    )  # 90 degrees about the x axis
    rotated_lattice = np.dot(lattice, rotation_matrix)
    return rotated_lattice

def train_val_split_indices(total_size, val_ratio=0.2, random_seed=184):
    np.random.seed(random_seed)
    indices = np.arange(total_size)
    np.random.shuffle(indices)
    split = int(np.floor(val_ratio * total_size))
    val_indices = indices[:split]
    train_indices = indices[split:]
    return train_indices, val_indices

def main():
    NUM_FILES = 5 # TODO: increase

    atomic_numbers = []
    lattice = []
    frac_x = []
    idx_start = []
    num_atoms = []
    for i in range(NUM_FILES):
        file_name = f"alexandria_ps_00{i}"
        atomic_numbers_i, lattice_i, frac_x_i, idx_start_i, num_atoms_i = prep_data(file_name)
        atomic_numbers.append(atomic_numbers_i)
        lattice.append(lattice_i)
        frac_x.append(frac_x_i)
        idx_start.append(idx_start_i)
        num_atoms.append(num_atoms_i)

    atomic_numbers = np.concatenate(atomic_numbers, axis=0)
    lattice = np.concatenate(lattice, axis=0)
    frac_x = np.concatenate(frac_x, axis=0)
    idx_start = np.concatenate(idx_start, axis=0)
    num_atoms = np.concatenate(num_atoms, axis=0)

    num_rows = idx_start.shape[0]
    val_ratio = 0.02 # TODO: use a legit train/val split
    train_indices, val_indices = train_val_split_indices(num_rows, val_ratio)

    def get_train_val(arr):
        train_arr = np.concatenate([arr[idx_start[i]:idx_start[i]+num_atoms[i]] for i in train_indices], axis=0)
        val_arr =  np.concatenate([arr[idx_start[i]:idx_start[i]+num_atoms[i]] for i in val_indices], axis=0)
        return train_arr, val_arr


    train_all_atomic, val_all_atomic = get_train_val(atomic_numbers)

    train_all_lattice = [lattice[i] for i in train_indices]
    val_all_lattice = [lattice[i] for i in val_indices]

    train_all_frac_coords, val_all_frac_coords = get_train_val(frac_x)

    train_all_num_atoms = [num_atoms[i] for i in train_indices]
    val_all_num_atoms = [num_atoms[i] for i in val_indices]

    train_all_idx_start = []
    start_idx = 0
    for atom_cnt in train_all_num_atoms:
        train_all_idx_start.append(start_idx)
        start_idx += atom_cnt

    val_all_idx_start = []
    start_idx = 0
    for atom_cnt in val_all_num_atoms:
        val_all_idx_start.append(start_idx)
        start_idx += atom_cnt

    save_dataset("train_all", train_all_atomic, train_all_lattice, train_all_frac_coords, train_all_idx_start, train_all_num_atoms)
    save_dataset("val_all", val_all_atomic, val_all_lattice, val_all_frac_coords, val_all_idx_start, val_all_num_atoms)

    idx = num_atoms[:10].sum()
    save_dataset("train_10", atomic_numbers[:idx], lattice[:10], frac_x[:idx], idx_start[:10], num_atoms[:10])
    save_dataset("val_10", atomic_numbers[:idx], rotate_lattice_about_origin(lattice[:10]), frac_x[:idx], idx_start[:10], num_atoms[:10])

    idx = num_atoms[0]
    save_dataset("train_1", atomic_numbers[:idx], lattice[:1], frac_x[:idx], idx_start[:1], num_atoms[:1])
    save_dataset("val_1", atomic_numbers[:idx], rotate_lattice_about_origin(lattice[:1]), frac_x[:idx], idx_start[:1], num_atoms[:1])

if __name__ == "__main__":
    main()