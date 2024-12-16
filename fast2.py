import torch

from create_graph import NUM_OFFSETS, OFFSETS, _compute_img_positions_torch, points_in_parallelepiped
from fast_approach import extend_lattice


def get_sub_parallelepiped_for_supercell_side(lattice: torch.Tensor, supercell_side: torch.Tensor, radius: int):
    lengths = torch.linalg.norm(lattice, axis=1)
    unit_vectors = lattice / lengths[:, None]
    extended_lattice = lattice + radius * unit_vectors
    extended_lattice = extended_lattice - (radius * unit_vectors//2)
    return extended_lattice

def _compute_img_positions_torch2(
    positions: torch.Tensor, periodic_boundaries: torch.Tensor
) -> torch.Tensor:
    """Computes the positions of the periodic images of the input structure.

    Consider the following 2D periodic boundary image.
    + --- + --- + --- +
    |     |     |     |
    + --- + --- + --- +
    |     |  x  |     |
    + --- + --- + --- +
    |     |     |     |
    + --- + --- + --- +

    Each tile in this has an associated translation to translate
    'x'. For example, the top left would by (-1, +1). These are
    the 'OFFSETS', but OFFSETS are for a 3x3x3 grid.

    This is complicated by the fact that our periodic
    boundaries are not orthogonal to each other, and so we form a new
    translation by taking a linear combination of the unit cell axes.

    Args:
        positions (torch.Tensor): Positions of the atoms. Shape [num_atoms, 3].
        periodic_boundaries (torch.Tensor): Periodic boundaries of the unit cell.
            This can be 2 shapes - [3, 3] or [num_atoms, 3, 3]. If the shape is
            [num_atoms, 3, 3], it is assumed that the PBC has been repeat_interleaved
            for each atom, i.e this function is agnostic as to whether it is computing
            with respect to a batch or not.
    Returns:
        torch.Tensor: The positions of the periodic images. Shape [num_atoms, 27, 3].
    """
    num_positions = len(positions)

    has_unbatched_pbc = periodic_boundaries.shape == (3, 3)
    if has_unbatched_pbc:
        periodic_boundaries = periodic_boundaries.unsqueeze(0)
        periodic_boundaries = periodic_boundaries.expand(num_positions, 3, 3)

    # This section *assumes* we have already repeat_interleaved the periodic
    # boundaries to be the same size as the positions. e.g:
    # (batch_size, 3, 3) -> (batch_n_node, 3, 3)
    assert periodic_boundaries.shape[0] == positions.shape[0]
    # First, create a tensor of offsets where the first axis
    # is the number of particles
    # Shape (27, 3)
    offsets = torch.tensor(OFFSETS, device=positions.device, dtype=positions.dtype)
    # Shape (1, 27, 3)
    offsets = torch.unsqueeze(offsets, 0)
    # Shape (batch_n_node, 27, 3)
    repeated_offsets = offsets.expand(num_positions, NUM_OFFSETS, 3)
    # offsets is now size (batch_n_node, 27, 3). Now we want a translation which is
    # a linear combination of the pbcs which is currently shape (batch_n_node, 3, 3).
    # Make offsets shape (batch_n_node, 27, 3, 1)
    repeated_offsets = torch.unsqueeze(repeated_offsets, 3)
    # Make pbcs shape (batch_n_node, 1, 3, 3)
    periodic_boundaries = torch.unsqueeze(periodic_boundaries, 1)
    # Take the linear combination.
    # Shape (batch_n_node, 27, 3, 3)
    translations = repeated_offsets * periodic_boundaries
    # Shape (batch_n_node, 27, 3)
    translations = translations.sum(2)

    # Expand the positions so we can broadcast add the translations per PBC image.
    # Shape (batch_n_node, 1, 3)
    expanded_positions = positions.unsqueeze(1)
    # Broadcasted addition. Shape (batch_n_node, 27, 3)
    translated_positions = expanded_positions + translations
    return translated_positions

def fast(*, lattice: torch.Tensor, frac_coord: torch.Tensor, radius: int = 5, max_number_neighbors: int, knn_library: str, n_workers: int = 1):
    cart_coords = frac_coord @ lattice

    # radius_for_side
    supercell_positions = _compute_img_positions_torch2(
        positions=cart_coords, periodic_boundaries=lattice
    )

    lattice2 = extend_lattice(lattice, radius)

    # the simplest way: just make a larger parallelepiped, then cross product each point with the larger parallelpied, and keep the points with a smaller cross product
    lattice2_mask = points_in_parallelepiped(lattice2, supercell_positions)
    supercell2_positions = torch.masked_select(supercell_positions, lattice2_mask.unsqueeze(-1))
