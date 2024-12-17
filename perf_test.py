import torch
from prep_datasets import load_dataset
from create_graph import compute_pbc_radius_graph
from fast4 import fast4
import time


def test_method():
    for i, config in enumerate(dataset):
        frac_coord = torch.tensor(config.frac_coord, dtype=torch.float32)
        lattice = torch.tensor(config.lattice, dtype=torch.float32)

        cart_coord = frac_coord @ lattice
        yield cart_coord, lattice

if __name__ == "__main__":
    torch.set_printoptions(sci_mode=False)  # Disable scientific notation
    knn_library = "pynanoflann"
    dataset = load_dataset("datasets/alexandria_hdf5/train_all.h5")

    radius = 5
    max_number_neighbors = 20

    # Timing the first method
    start_time = time.time()
    for cart_coord, lattice in test_method():
        edges1, displacements1 = compute_pbc_radius_graph(
            positions=cart_coord,
            periodic_boundaries=lattice,
            radius=radius,
            max_number_neighbors=max_number_neighbors,
            brute_force=False,
            library=knn_library
        )
    end_time = time.time()
    print(f"compute_pbc_radius_graph took {end_time - start_time:.4f} seconds")

    # Timing the second method
    start_time = time.time()
    for cart_coord, lattice in test_method():
        edges2, displacements2 = fast4(
            lattice=lattice,
            cart_coord=cart_coord,
            radius=radius,
            max_number_neighbors=20,
        )
    end_time = time.time()
    print(f"fast4 took {end_time - start_time:.4f} seconds")