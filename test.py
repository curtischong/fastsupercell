import numpy as np
import h5py
from dataclasses import dataclass
import os

@dataclass
class Configuration:
    # fields with a prefix "prev_" are the real target value. These are optional since during inference, we don't know them
    atomic_numbers: np.ndarray  # this is a list of atomic numbers. NOT one-hot-encoded
    frac_coord: np.ndarray
    lattice: np.ndarray


def load_data(filepath: str):
    # assert the file exists
    assert os.path.isfile(filepath)

    with h5py.File(filepath, "r") as file:
        frac_x=file["crystals"]["frac_x"][:],
        atomic_numbers=file["crystals"]["atomic_numbers"][:],
        lattice=file["crystals"]["lattice"][:],
        num_atoms=file["crystals"]["num_atoms"][:],
        idx_start=file["crystals"]["idx_start"][:],

    return frac_x[0], atomic_numbers[0], lattice[0], num_atoms[0], idx_start[0]


def load_dataset(file_path) -> list[Configuration]:
    frac_x, atomic_numbers, lattice, num_atoms, idx_start = load_data(file_path)

    dataset = []
    for i in range(lattice.shape[0]):
        start = idx_start[i]
        end = start + num_atoms[i]
        config = Configuration(
            atomic_numbers=atomic_numbers[start:end],
            frac_coord=frac_x[start:end],
            lattice=lattice[i],
        )
        dataset.append(config)
    return dataset

if __name__ == "__main__":
    dataset = load_dataset("datasets/alexandria_hdf5/train_10.h5")
    print(dataset)