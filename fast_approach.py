# can we use a linear transformation so instead of a parallelepiped, we create the graph in a euclidian plane?
# this way, when calculating the distances between two atoms, we can use a bitmask
# maybe diff axis will have diff "length" cause of the mapped distances
# now we can generate the edges VERY fast
# a future optimization would be to use sram instead

import numpy as np
from create_graph import NUM_OFFSETS, _compute_img_positions_torch, points_in_parallelepiped, positions_to_graph
import torch
from pynanoflann import KDTree as NanoKDTree
from scipy.spatial import KDTree as SciKDTree

# we transform all the points to a normal 1x1x1 cube
# then we just find the points near the edges of that cube
# lastly we map those points to the hypercube and do the kd tree calculation for edges


# https://shad.io/MatVis/

def normal_vectors(lattice):
    """
    Given a 3x3 lattice matrix (rows are vectors a, b, c),
    return the normal vectors to each plane.
    """
    a = lattice[0]
    b = lattice[1]
    c = lattice[2]

    n_a = np.cross(b, c)  # normal to face opposite a
    n_b = np.cross(c, a)  # normal to face opposite b
    n_c = np.cross(a, b)  # normal to face opposite c

    return n_a, -n_a, n_b, -n_b, n_c, -n_c

def extend_lattice(lattice, radius):
    lengths = torch.linalg.norm(lattice, axis=1)
    unit_vectors = lattice / lengths[:, None]

    additional_lengths = (radius * unit_vectors)
    extended_lattice = lattice + additional_lengths
    position_offset = torch.sum(-additional_lengths/2, dim=0) # dim=0 cause we want to sum up all the contributions along the x-axis (for example)
    return extended_lattice, position_offset

def extend_lattice2(lattice, radius):
    lengths = torch.linalg.norm(lattice, axis=1)
    unit_vectors = lattice / lengths[:, None]

    additional_lengths = (radius * unit_vectors)
    extended_lattice = lattice + additional_lengths
    position_offset = torch.sum(-additional_lengths/2, dim=0) # dim=0 cause we want to sum up all the contributions along the x-axis (for example)
    return extended_lattice, position_offset


def fast(*, lattice: torch.Tensor, frac_coord: torch.Tensor, radius: int, max_number_neighbors: int, knn_library: str, n_workers: int = 1):

    frac_coord = frac_coord
    cart_coord = frac_coord @ lattice

    cart_supercell_coords = _compute_img_positions_torch(cart_coord, lattice)
    cart_supercell_coords = cart_supercell_coords.reshape(-1, 3)

    extended_lattice, position_offset = extend_lattice(lattice, radius)

    lattice2_mask = points_in_parallelepiped(extended_lattice, position_offset, cart_supercell_coords)
    supercell2_positions = torch.masked_select(cart_supercell_coords, lattice2_mask.unsqueeze(-1)) # these are the relevant positions of the supercell (the ones closest to the lattice)

    num_positions = len(cart_coord)
    node_id = torch.arange(num_positions).unsqueeze(-1).expand(num_positions, NUM_OFFSETS)

    # the simplest way: just make a larger parallelepiped, then cross product each point with the larger parallelpied, and keep the points with a smaller cross product
    node_id2 = torch.masked_select(node_id.reshape(-1), lattice2_mask)


    return masked_positions_to_graph(supercell2_positions.reshape(-1, 3), positions=cart_coord, node_id2=node_id2, radius=radius, max_number_neighbors=max_number_neighbors, n_workers=n_workers)


def masked_positions_to_graph(supercell_positions, positions, node_id2, radius, max_number_neighbors, n_workers):
    tree_data = supercell_positions.clone().detach().cpu().numpy()
    tree_query = positions.clone().detach().cpu().numpy()
    num_positions = positions.shape[0]

    tree = NanoKDTree(
        n_neighbors=min(max_number_neighbors + 1, len(supercell_positions)),
        radius=radius,
        leaf_size=100,
        metric="l2",
    )
    tree.fit(tree_data)
    distance_values, nearest_img_neighbors = tree.kneighbors(
        tree_query, n_jobs=n_workers
    )
    nearest_img_neighbors = nearest_img_neighbors.astype(np.int32)  # type: ignore

    # remove the self node which will be closest
    index_array = nearest_img_neighbors[:, 1:]
    # remove distances greater than radius
    within_radius = distance_values[:, 1:] < (radius + 1e-6)
    receivers_imgs = index_array[within_radius]
    num_neighbors_per_position = within_radius.sum(-1)

    # We construct our senders and receiver indexes.
    senders = np.repeat(np.arange(num_positions), list(num_neighbors_per_position))  # type: ignore
    receivers_img_torch = torch.tensor(receivers_imgs, device=positions.device)
    # Map back to indexes on the central image.
    # receivers = receivers_img_torch % num_positions # this no longer works since we pruned the supercell
    receivers = node_id2[receivers_img_torch]
    senders_torch = torch.tensor(senders, device=positions.device)

    # Finally compute the vector displacements between senders and receivers.
    vectors = supercell_positions[receivers_img_torch] - positions[senders_torch]
    return torch.stack((senders_torch, receivers), dim=0), vectors

# strat:
    # 1. use knn and get all the edges without PBC
    # 2. manually match each of the faces with each other. the nodes can only connect to the other nodes on the other side of the plane

    # how do I do this?
    # we can just create a knn for each node
    # screw it. I feel like the supercell appraoch is still the best cause we can ues the knn logic

    # is it faster to create the 27 cube. or we just append extra nodes?


    # in the original euclidiean cube, we find all the nodes that are near the face.
    # then we just add it to the coords later. finally we use the kd tree

    # in general, I think that drastically reducing the number of nodes is best.


    # how do I know the radius for the face?


    # we don't even need tod do the transofrmation. we can just slide the lattice plane and remove coords that fall outside the plane (after we get the suepercells)