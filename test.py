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
    edgest = edges.transpose(0, 1)
    for i in range(edges.shape[1]):
        edge = edgest[i]
        displacement = displacements[i]
        tuples.append((edge[0].item(), edge[1].item(), displacement[0].item(), displacement[1].item(), displacement[2].item()))
        # one bug that happened was that I noticed that the displacements WERE THE SAME between my implementation and the orb implementation
        # it meant that I got the distances right. I just mapped the wrong node IDs. So I did this to verify that only the node IDs were wrong (using the below code passed the test!):
        # tuples.append((displacement[0].item(), displacement[1].item(), displacement[2].item()))
    return sorted(tuples)

def graphs_are_equal(edges1, edges2, displacements1, displacements2):
    assert edges1.shape == edges2.shape
    assert displacements1.shape == displacements2.shape
    tuples1 = edges_to_tuples(edges1, displacements1)
    tuples2 = edges_to_tuples(edges2, displacements2)
    print("tuples1")
    for t in tuples1:
        print(t)
    print("tuples2")
    for t in tuples2:
        print(t)
    for tuple1, tuple2 in zip(tuples1, tuples2):
        for i in range(len(tuple1)):
            assert tuple1[i] == tuple2[i], f"{tuple1} != {tuple2}"


if __name__ == "__main__":
    torch.set_printoptions(sci_mode=False)  # Disable scientific notation
    knn_library = "pynanoflann"
    dataset = load_dataset("datasets/alexandria_hdf5/train_10.h5")
    for i, config in enumerate(dataset):
        frac_coord = torch.tensor(config.frac_coord, dtype=torch.float32)
        lattice = torch.tensor(config.lattice, dtype=torch.float32)

        cart_coord = frac_coord @ lattice
        atomic_numbers = torch.tensor(config.atomic_numbers, dtype=torch.int64)
        print(f"frac_coord: {frac_coord}")
        print(f"lattice: {lattice}")
        print(f"atomic_numbers: {atomic_numbers}")

        radius=5

        edges1, displacements1 = compute_pbc_radius_graph(
            positions=cart_coord,
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