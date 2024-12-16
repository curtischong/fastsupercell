import numpy as np
from create_graph import compute_pbc_radius_graph
from fast3 import fast3
from fast_approach import fast
from prep_datasets import load_dataset
import torch


if __name__ == "__main__":
    knn_library = "pynanoflann"
    dataset = load_dataset("datasets/alexandria_hdf5/train_10.h5")
    # print(dataset)
    for i, config in enumerate(dataset):
        frac_coord = torch.tensor(config.frac_coord, dtype=torch.float32)
        lattice = torch.tensor(config.lattice, dtype=torch.float32)
        atomic_numbers = torch.tensor(config.atomic_numbers, dtype=torch.int64)
        print(f"frac_coord: {frac_coord}")
        print(f"lattice: {lattice}")
        print(f"atomic_numbers: {atomic_numbers}")

        radius=1.0

        edges1, displacements1 = compute_pbc_radius_graph(
            positions=frac_coord,
            periodic_boundaries=lattice,
            radius=radius,
            max_number_neighbors=20,
            brute_force=False,
            library=knn_library
        )

        edges2, displacements2 = fast3(
            lattice=lattice,
            frac_coord=frac_coord,
            radius=radius,
            max_number_neighbors=20,
            knn_library=knn_library
        )

        # assert edges and displacements are the same
        assert edges1.shape == edges2.shape
        assert np.allclose(edges1, edges2)
        assert np.allclose(displacements1, displacements2)
        print(f"success: {i}")
    print("we can caculate the graph properly!")