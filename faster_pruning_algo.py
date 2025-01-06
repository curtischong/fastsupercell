import numpy as np
from create_graph import NUM_OFFSETS, _compute_img_positions_torch, points_in_parallelepiped, positions_to_graph
import torch
from pynanoflann import KDTree as NanoKDTree
from scipy.spatial import KDTree as SciKDTree



def smarter_calc_distance(coords: list[float], lattice: list[list[float]]):
    n = len(coords)
    for i in range(n):
        coord = coords[i]

        pass

# https://shad.io/MatVis/

def compute_normals(lattice: torch.Tensor, radius: int):
    """
    Compute the normal vectors to the three lattice planes defined
    by a set of three lattice vectors.

    Parameters
    ----------
    lattice : torch.Tensor
        A 3x3 tensor representing three lattice vectors.
        By convention, we assume each row of 'lattice' is one lattice vector.
        For example:
        lattice = torch.tensor([
            [a_x, a_y, a_z],
            [b_x, b_y, b_z],
            [c_x, c_y, c_z]
        ], dtype=torch.float)

    Returns
    -------
    normals : torch.Tensor
        A 3x3 tensor where each row is a normal vector corresponding
        to the plane formed by the other two lattice vectors.
        Specifically, the returned tensor has:
        normals[0] = b × c
        normals[1] = c × a
        normals[2] = a × b
    """

    # Extract the lattice vectors a, b, c
    a = lattice[0]
    b = lattice[1]
    c = lattice[2]

    # Compute the normals using cross products
    n_a = torch.linalg.cross(b, c)  # normal to plane defined by b and c
    n_b = torch.linalg.cross(c, a)  # normal to plane defined by c and a
    n_c = torch.linalg.cross(a, b)  # normal to plane defined by a and b

    # Stack the normals into a single tensor
    normals = torch.stack([n_a, n_b, n_c], dim=0)

    # normalize so it's all of length radius
    normals /= torch.linalg.norm(normals, dim=1)
    normals *= radius

    return normals

def create_masking_parallelepiped(lattice: torch.Tensor, radius: int):
    normals = compute_normals(lattice, radius)
    inverse_lattice = torch.linalg.inv(lattice)
    inverted_normals = normals @ inverse_lattice
    inverted_normals_norms = torch.linalg.norm(inverted_normals, dim=1)

    extended_cube = torch.ones(3) + 2*inverted_normals_norms

    extended_lattice = lattice * extended_cube 
    additional_lengths = extended_lattice - lattice
    position_offset = torch.sum(-additional_lengths/2, dim=0) # dim=0 cause we want to sum up all the contributions along the x-axis (for example)
    return extended_lattice, position_offset

def compute_pbc_radius_graph_with_pruning(*, lattice: torch.Tensor, cart_coord: torch.Tensor, radius: int = 5, max_number_neighbors: int, n_workers: int = 1):

    cart_supercell_coords = _compute_img_positions_torch(cart_coord, lattice)
    cart_supercell_coords = cart_supercell_coords.transpose(0, 1)
    cart_supercell_coords = cart_supercell_coords.reshape(-1, 3)

    extended_lattice, position_offset = create_masking_parallelepiped(lattice, radius)

    lattice2_mask = points_in_parallelepiped(extended_lattice, position_offset, cart_supercell_coords)
    supercell2_positions = torch.masked_select(cart_supercell_coords, lattice2_mask.unsqueeze(-1)) # these are the relevant positions of the supercell (the ones closest to the lattice)

    num_positions = len(cart_coord)
    node_id = torch.arange(num_positions).unsqueeze(-1).expand(num_positions, NUM_OFFSETS).transpose(0, 1)
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
