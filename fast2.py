from create_graph import NUM_OFFSETS, OFFSETS, _compute_img_positions_torch, points_in_parallelepiped
import torch
from fast_approach import masked_positions_to_graph


def extend_lattice2(lattice, radius):
    lengths = torch.linalg.norm(lattice, axis=1)
    unit_vectors = lattice / lengths[:, None]

    additional_lengths = (radius * unit_vectors)
    extended_lattice = lattice + additional_lengths
    position_offset = torch.sum(-additional_lengths/2, dim=0) # dim=0 cause we want to sum up all the contributions along the x-axis (for example)
    return extended_lattice, position_offset

# OFFSETS

def create_mask(num_atoms: int, radius: torch.Tensor, lattice: torch.Tensor):
    leq_mask = torch.zeros(27, 3)
    coord_mul = torch.zeros(27, 3)
    for i in range(NUM_OFFSETS):
        offset = OFFSETS[i]

        translated = lattice + offset

        for j in range(3):
            offset_i = offset[j]
            if offset_i < 0:
                coord_mul[i][j] = -1
                leq_mask[i][j] = -(1 - radius_in_frac_amount)
            elif offset_i > 0:
                coord_mul[i][j] = 1
                leq_mask[i][j] = radius_in_frac_amount
            

    leq_mask = leq_mask.expand(num_atoms, 27, 3)
    coord_mul = coord_mul.expand(num_atoms, 27, 3)
    return coord_mul, leq_mask

def fast2(*, lattice: torch.Tensor, frac_coord: torch.Tensor, radius: int = 5, max_number_neighbors: int, knn_library: str, n_workers: int = 1):

    frac_coord = frac_coord
    cart_coord = frac_coord @ lattice

    cart_supercell_coords = _compute_img_positions_torch(frac_coord, lattice)


    coord_mul, leq_mask = create_mask(frac_coord.shape[0], radius, lattice)
    mask = torch.less_equal(coord_mul * cart_supercell_coords, leq_mask)
    # cart_supercell_coords = cart_supercell_coords.reshape(-1, 3)

    extended_lattice, position_offset = extend_lattice(lattice, radius)

    lattice2_mask = points_in_parallelepiped(extended_lattice, position_offset, cart_supercell_coords)
    supercell2_positions = torch.masked_select(cart_supercell_coords, lattice2_mask.unsqueeze(-1)) # these are the relevant positions of the supercell (the ones closest to the lattice)

    num_positions = len(cart_coord)
    node_id = torch.arange(num_positions).unsqueeze(-1).expand(num_positions, NUM_OFFSETS)

    # the simplest way: just make a larger parallelepiped, then cross product each point with the larger parallelpied, and keep the points with a smaller cross product
    node_id2 = torch.masked_select(node_id.reshape(-1), lattice2_mask)


    return masked_positions_to_graph(supercell2_positions.reshape(-1, 3), positions=cart_coord, node_id2=node_id2, radius=radius, max_number_neighbors=max_number_neighbors, n_workers=n_workers)
