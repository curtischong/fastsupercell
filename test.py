import numpy as np
from create_graph import compute_pbc_radius_graph
from pbc_kdtree import compute_pbc_radius_graph_using_kd_tree
from pruning_algo import compute_pbc_radius_graph_with_pruning
from prep_datasets import load_dataset
import torch

def edges_to_tuples(edges, displacements):
    assert edges.shape[1] == displacements.shape[0], f"{edges.shape}, {displacements.shape}"
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


# here's how we would make a better error metric to compare how similar two diff knn methods are:
# 1. sort by displacements. make sure that we have nearly the same number of neighbors that are near. it's okay to have different amounts of neighbors that are far (closer to the cutoff point)
#    - This is because in NNPs, there is a radial cutoff function. so the further the neighbor, the less their contribution is during message passing
#    - this means that the number of neighbors don't matter
# 2. for edge errors that are not the EXACT same, check to see if it's the same displacement. If it is, it's okay. it shouldn't contribute much to the err between the two knn methods
def graphs_are_equal(edges1, edges2, displacements1, displacements2, ith_graph):
    assert edges1.shape == edges2.shape, f"edges1: {edges1.shape}, edges2: {edges2.shape}"
    assert displacements1.shape == displacements2.shape
    tuples1 = set(edges_to_tuples(edges1, displacements1))
    tuples2 = set(edges_to_tuples(edges2, displacements2))

    symmetric_difference = tuples1.symmetric_difference(tuples2)
    num_mismatch = len(symmetric_difference)
    set_size = len(tuples1)
    err_rate = num_mismatch / (2*set_size)
    if err_rate > 0.2:
        print("tuples1")
        for t in tuples1:
            print(t)
        print("tuples2")
        for t in tuples2:
            print(t)
        print(f"ith_graph: ", ith_graph)
        print(f"num_mismatch: {num_mismatch}")
        print(f"set_size: {set_size}")
        print(f"err_rate: {err_rate}")
        exit(1)
    return err_rate


if __name__ == "__main__":
    torch.set_printoptions(sci_mode=False)  # Disable scientific notation
    knn_library = "pynanoflann"
    dataset = load_dataset("datasets/alexandria_hdf5/train_all.h5")
    for i, config in enumerate(dataset):
        frac_coord = torch.tensor(config.frac_coord, dtype=torch.float32)
        lattice = torch.tensor(config.lattice, dtype=torch.float32)

        cart_coord = frac_coord @ lattice

        radius=5


        edges2, displacements2 = compute_pbc_radius_graph_using_kd_tree(
            lattice=lattice,
            cart_coord=cart_coord,
            radius=radius,
            max_number_neighbors=20,
        )

        edges1, displacements1 = compute_pbc_radius_graph(
            positions=cart_coord,
            periodic_boundaries=lattice,
            radius=radius,
            max_number_neighbors=20,
            brute_force=False,
            library=knn_library
        )

        err = graphs_are_equal(edges1, edges2, displacements1, displacements2, i)
        print(f"success: {i}. err={err}")
    print("we can calculate the graph properly!")