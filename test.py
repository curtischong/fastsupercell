import numpy as np
from create_graph import compute_pbc_radius_graph
from fast3 import fast3
from fast4 import fast4
from fast_approach import fast
from prep_datasets import load_dataset
import torch

def edges_to_tuples(edges, displacements):
    assert edges.shape[1] == displacements.shape[0]
    tuples = []
    for i in range(len(edges)):
        edge = edges[i]
        displacement = displacements[i]
        tuples.append((edge[0], edge[1], displacement)) # TODO: make displacement a tuple so we can compare (and sort it)
    return sorted(tuples)

def graphs_are_equal(edges1, edges2, displacements1, displacements2):
    assert edges1.shape == edges2.shape
    tuples1 = edges_to_tuples(edges1, displacements1)
    tuples2 = edges_to_tuples(edges2, displacements2)
    for tuple1, tuple2 in zip(tuples1, tuples2):
        assert tuple1[0] == tuple2[0] and tuple1[1] == tuple2[1] and torch.allclose(tuple1[2], tuple2[2]), f"{tuple1} != {tuple2}"


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

        radius=10.0

        edges1, displacements1 = compute_pbc_radius_graph(
            positions=frac_coord,
            periodic_boundaries=lattice,
            radius=radius,
            max_number_neighbors=20,
            brute_force=False,
            library=knn_library
        )

        edges2, displacements2 = fast4(
            lattice=lattice,
            frac_coord=frac_coord,
            radius=radius,
            max_number_neighbors=20,
            knn_library=knn_library
        )

        graphs_are_equal(edges1, edges2, displacements1, displacements2)
        print(f"success: {i}")
    print("we can calculate the graph properly!")