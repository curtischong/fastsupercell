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



def prep_data_and_save_hdf5(filename, take_max_num_examples=None):
    print(f"prepping {filename}")
    with bz2.open(f"{IN_DIR}/{filename}.json.bz2", "rt", encoding="utf-8") as fh:
        data = json.load(fh)
    if take_max_num_examples is not None:
        filename = f"{filename}_take{take_max_num_examples}"
        data["entries"] = data["entries"][:take_max_num_examples]

    entries = [ComputedStructureEntry.from_dict(i) for i in data["entries"]]
    print(f"Found {len(entries)} entries for {filename}")

    atomic_number_vectors = []
    lattice_matrices = np.zeros((len(entries), 3, 3))
    frac_coords_arrays = []
    for idx, entry in enumerate(entries):
        structure = entry.structure
        atomic_number_vector = np.empty(len(structure.species), dtype=int)
        for i, species in enumerate(structure.species):
            atomic_number_vector[i] = species.Z
        atomic_number_vectors.append(atomic_number_vector)

        # new_lattice = normalize_lattice(lattice)

        lattice_matrices[idx] = structure.lattice.matrix
        frac_coords_arrays.append(structure.frac_coords)

    save_dataset(filename, atomic_number_vectors, lattice_matrices, frac_coords_arrays)


def save_dataset(filename, atomic_number_vectors, lattice_matrices, frac_coords_arrays):
    # Save the data to an HDF5 file
    os.makedirs(OUT_DIR, exist_ok=True)
    with h5py.File(f"{OUT_DIR}/{filename}.h5", "w") as f:
        atom_onehot_group = f.create_group("atomic_number")
        for i, vector in enumerate(atomic_number_vectors):
            atom_onehot_group.create_dataset(str(i), data=vector)

        f.create_dataset("lattice_matrix", data=lattice_matrices)

        frac_coords_group = f.create_group("frac_coord")
        for i, array in enumerate(frac_coords_arrays):
            frac_coords_group.create_dataset(str(i), data=array)


def prep_10_examples():
    prep_data_and_save_hdf5("alexandria_ps_000", take_max_num_examples=10)


def main():
    prep_10_examples()  # this prepares a small sample dataset so when we train locally, it's fast.

    NUM_FILES = 5

    processes = []
    for i in range(NUM_FILES):
        file_name = f"alexandria_ps_00{i}"

        # https://stackoverflow.com/questions/55529319/how-to-create-multiple-threads-dynamically-in-python
        p = Process(target=prep_data_and_save_hdf5, args=(file_name,))
        p.start()
        processes.append(p)

    # Wait all processes to finish.
    for p in processes:
        p.join()


if __name__ == "__main__":
    main()
