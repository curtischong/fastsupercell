from create_graph import NUM_OFFSETS, OFFSETS, _compute_img_positions_torch, points_in_parallelepiped
import torch
from fast_approach import masked_positions_to_graph


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
    n_a = torch.cross(b, c)  # normal to plane defined by b and c
    n_b = torch.cross(c, a)  # normal to plane defined by c and a
    n_c = torch.cross(a, b)  # normal to plane defined by a and b

    # Stack the normals into a single tensor
    normals = torch.stack([n_a, n_b, n_c], dim=0)

    # normalize so it's all of length radius
    normals /= torch.linalg.norm(normals, dim=1)
    normals *= radius

    return normals

def create_mask(frac_coords: torch.Tensor, radius: torch.Tensor, lattice: torch.Tensor):
    normals = compute_normals(lattice, radius)
    inverse_lattice = torch.linalg.inv(lattice)
    inverted_normals = normals @ inverse_lattice
    inverted_normal_norms = torch.linalg.norm(inverted_normals, dim=1)

    # no need for inverted farc.s it's laready frac. cause its a unit square
    # inverted_fracs = frac_coords @ inverse_lattice


    num_coords = frac_coords.shape[0]
    masked_coords = torch.zeros(num_coords, 27, dtype=torch.bool)
    for a in range(NUM_OFFSETS):
        offset = OFFSETS[a]
        for b in range(num_coords):
            frac_coord = frac_coords[b]
            is_in_range = True
            for c in range(3):
                offset_c = offset[c]
                cutoff_norm_amount = inverted_normal_norms[c]
                if offset_c < 0:
                    is_in_range = is_in_range and frac_coord[c] >= 1 - cutoff_norm_amount
                elif offset_c > 0:
                    is_in_range = is_in_range and frac_coord[c] <= cutoff_norm_amount

            masked_coords[b][a] = is_in_range
    return masked_coords

def fast3(*, lattice: torch.Tensor, frac_coord: torch.Tensor, radius: int = 5, max_number_neighbors: int, knn_library: str, n_workers: int = 1):

    frac_coord = frac_coord
    cart_coord = frac_coord @ lattice

    cart_supercell_coords = _compute_img_positions_torch(frac_coord, lattice)


    masked_coords = create_mask(frac_coord, radius, lattice)
    supercell2_positions = torch.masked_select(cart_supercell_coords, masked_coords.unsqueeze(-1)) # these are the relevant positions of the supercell (the ones closest to the lattice)

    num_positions = len(cart_coord)
    node_id = torch.arange(num_positions).unsqueeze(-1).expand(num_positions, NUM_OFFSETS)

    # the simplest way: just make a larger parallelepiped, then cross product each point with the larger parallelpied, and keep the points with a smaller cross product
    node_id2 = torch.masked_select(node_id, masked_coords)


    return masked_positions_to_graph(supercell2_positions.reshape(-1, 3), positions=cart_coord, node_id2=node_id2, radius=radius, max_number_neighbors=max_number_neighbors, n_workers=n_workers)
