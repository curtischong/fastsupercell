import torch
import kdtree_3torus



def compute_pbc_radius_graph_using_kd_tree(*, lattice: torch.Tensor, cart_coord: torch.Tensor, radius: int = 5, max_number_neighbors: int):
    res = kdtree_3torus.create_nn_graph_np(coords=cart_coord, lattice=lattice, radius=radius, max_neighbors=max_number_neighbors, exclude_self=True)
    return res["edges"], res["disp"]
