import torch
from typing import Union, Optional, Tuple
import numpy as np
from scipy.spatial import KDTree as SciKDTree
from pynanoflann import KDTree as NanoKDTree

# These are offsets applied to coordinates to create a 3x3x3
# tiled periodic image of the input structure.
OFFSETS = np.array(
    [
        [-1.0, 1.0, -1.0],
        [0.0, 1.0, -1.0],
        [1.0, 1.0, -1.0],
        [-1.0, 0.0, -1.0],
        [0.0, 0.0, -1.0],
        [1.0, 0.0, -1.0],
        [-1.0, -1.0, -1.0],
        [0.0, -1.0, -1.0],
        [1.0, -1.0, -1.0],
        [-1.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
        [1.0, 1.0, 0.0],
        [-1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [-1.0, -1.0, 0.0],
        [0.0, -1.0, 0.0],
        [1.0, -1.0, 0.0],
        [-1.0, 1.0, 1.0],
        [0.0, 1.0, 1.0],
        [1.0, 1.0, 1.0],
        [-1.0, 0.0, 1.0],
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 1.0],
        [-1.0, -1.0, 1.0],
        [0.0, -1.0, 1.0],
        [1.0, -1.0, 1.0],
    ]
)
NUM_OFFSETS = len(OFFSETS)

def get_device(
    requested_device: Optional[Union[torch.device, str, int]] = None
) -> torch.device:
    """Get a torch device, defaulting to gpu if available."""
    if requested_device is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested_device)


def points_in_parallelepiped(
    lattice: torch.Tensor, 
    lattice2_offset: torch.Tensor, 
    positions: torch.Tensor, 
    tol=1e-12
):
    """
    Check which positions lie inside the parallelepiped defined by the lattice vectors,
    taking into account an additional offset.

    Parameters
    ----------
    lattice : torch.Tensor of shape (3, 3)
        The lattice vectors defining the parallelepiped. 
        For example, lattice[0] = a, lattice[1] = b, lattice[2] = c.
    lattice2_offset : torch.Tensor of shape (3,)
        The translation offset to be applied to the parallelepiped.
    positions : torch.Tensor of shape (num_particles, 3)
        The positions of particles to test.
    tol : float, optional
        Tolerance for numerical comparisons.

    Returns
    -------
    inside_mask : torch.BoolTensor of shape (num_particles,)
        Boolean tensor where True indicates the corresponding particle 
        is inside or on the boundary of the shifted parallelepiped.
    """

    # Ensure inputs are in floating point format
    lattice = lattice.to(torch.float64)
    lattice2_offset = lattice2_offset.to(torch.float64)
    positions = positions.to(torch.float64)

    # Adjust positions by subtracting the offset so that
    # we can treat this as a standard parallelepiped check
    shifted_positions = positions - lattice2_offset

    # Extract lattice vectors
    a = lattice[0]
    b = lattice[1]
    c = lattice[2]

    # Compute volume: V = a · (b × c)
    bc_cross = torch.cross(b, c, dim=0)
    V = torch.dot(a, bc_cross)

    # If volume ~ 0, we have a degenerate parallelepiped
    if torch.abs(V) < tol:
        # No points are considered inside in a degenerate case
        return torch.zeros(positions.shape[0], dtype=torch.bool)

    # Precompute the other cross products
    ca_cross = torch.cross(c, a, dim=0)
    ab_cross = torch.cross(a, b, dim=0)

    # Compute u, v, w for each shifted position
    # u = ((P - offset) · (b × c)) / V
    # v = ((P - offset) · (c × a)) / V
    # w = ((P - offset) · (a × b)) / V
    u_values = (shifted_positions @ bc_cross) / V
    v_values = (shifted_positions @ ca_cross) / V
    w_values = (shifted_positions @ ab_cross) / V

    # Check if each of u, v, w are in [0,1] within tolerance
    inside_mask = (
        (u_values >= -tol) & (u_values <= 1 + tol) &
        (v_values >= -tol) & (v_values <= 1 + tol) &
        (w_values >= -tol) & (w_values <= 1 + tol)
    )

    return inside_mask

def _compute_img_positions_torch(
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

def brute_force_knn(
    img_positions: torch.Tensor, positions: torch.Tensor, k: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Brute force k-nearest neighbors.

    Args:
        img_positions (torch.Tensor): The positions of the images. Shape [num_atoms * 27, 3].
        positions (torch.Tensor): The positions of the query atoms. Shape [num_atoms, 3].
        k (int): The number of nearest neighbors to find.

    Returns:
        torch.return_types.topk: The indices of the nearest neighbors. Shape [num_atoms, k].
    """
    dist = torch.cdist(positions, img_positions)
    return torch.topk(dist, k, largest=False, sorted=True)


def positions_to_graph(supercell_positions:torch.Tensor, positions:torch.Tensor, radius:float, max_number_neighbors:int, library:str, n_workers:int):
    # Build a KDTree from the supercell positions.
    # Query that KDTree just for the positions in the central cell.
    tree_data = supercell_positions.clone().detach().cpu().numpy()
    tree_query = positions.clone().detach().cpu().numpy()
    distance_upper_bound = np.array(radius) + 1e-8
    num_positions = positions.shape[0]

    if library == "scipy":
        tree = SciKDTree(tree_data, leafsize=100)
        _, nearest_img_neighbors = tree.query(
            tree_query,
            max_number_neighbors + 1,
            distance_upper_bound=distance_upper_bound,
            workers=n_workers,
            p=2,
        )
        # Remove the self-edge that will be closest
        index_array = np.array(nearest_img_neighbors)[:, 1:]  # type: ignore
        # Remove any entry that equals len(supercell_positions), which are negative hits
        receivers_imgs = index_array[index_array != len(supercell_positions)]
        num_neighbors_per_position = (index_array != len(supercell_positions)).sum(
            -1
        )
    elif library == "pynanoflann":
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
    receivers = receivers_img_torch % num_positions
    senders_torch = torch.tensor(senders, device=positions.device)

    # Finally compute the vector displacements between senders and receivers.
    vectors = supercell_positions[receivers_img_torch] - positions[senders_torch]
    return torch.stack((senders_torch, receivers), dim=0), vectors

def compute_pbc_radius_graph(
    *,
    positions: torch.Tensor,
    periodic_boundaries: torch.Tensor,
    radius: Union[float, torch.Tensor],
    max_number_neighbors: int = 20,
    brute_force: Optional[bool] = None,
    library: str = "pynanoflann",
    n_workers: int = 1,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Computes periodic condition radius graph from positions.

    Args:
        positions (torch.Tensor): 3D positions of particles. Shape [num_particles, 3].
        periodic_boundaries (torch.Tensor): A 3x3 matrix where the periodic boundary axes are rows or columns.
        radius (Union[float, torch.tensor]): The radius within which to connect atoms.
        max_number_neighbors (int, optional): The maximum number of neighbors for each particle. Defaults to 20.
        brute_force (bool, optional): Whether to use brute force knn. Defaults to None, in which case brute_force
            is used if GPU is available (2-6x faster), but not on CPU (1.5x faster - 4x slower, depending on
            system size).
        library (str, optional): The KDTree library to use. Currently, either 'scipy' or 'pynanoflann'.
        n_workers (int, optional): The number of workers to use for KDTree construction. Defaults to 1.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: A 2-Tuple. First, an edge_index tensor, where the first index are the
        sender indices and the second are the receiver indices. Second, the vector displacements between edges.
    """
    device = get_device()
    if brute_force is None:
        # use brute force if positions are already on gpu
        brute_force = device.type != "cpu"

    if brute_force:
        # use gpu if available
        positions = positions.to(device)
        periodic_boundaries = periodic_boundaries.to(device)

    device = positions.device

    if torch.any(periodic_boundaries != 0.0):
        # Shape (num_positions, 27, 3)
        supercell_positions = _compute_img_positions_torch(
            positions=positions, periodic_boundaries=periodic_boundaries
        )
        # CRITICALLY IMPORTANT: We need to reshape the supercell_positions to be
        # flat, so we can use them for the nearest neighbors. The *way* in which
        # they are flattened is important, because we need to be able to map the
        # indices returned from the nearest neighbors to the original positions.
        # The easiest way to do this is to transpose, so that when we flatten, we
        # have:
        # [
        #   img_0_atom_0,
        #   img_0_atom_1,
        #   ...,
        #   img_0_atom_N,
        #   img_1_atom_0,
        #   img_1_atom_1,
        #   ...,
        #   img_N_atom_N,
        #   etc
        # ]
        # This way, we can take the mod of the indices returned from the nearest
        # neighbors to get the original indices.
        # Shape (27, num_positions, 3)
        supercell_positions = supercell_positions.transpose(0, 1)
        supercell_positions = supercell_positions.reshape(-1, 3)
    else:
        supercell_positions = positions

    num_positions = positions.shape[0]

    if brute_force:
        # Brute force
        distance_values, nearest_img_neighbors = brute_force_knn(
            supercell_positions,
            positions,
            min(max_number_neighbors + 1, len(supercell_positions)),
        )

        # remove distances greater than radius, and exclude self
        within_radius = distance_values[:, 1:] < (radius + 1e-6)

        num_neighbors_per_position = within_radius.sum(-1)
        # remove the self node which will be closest
        index_array = nearest_img_neighbors[:, 1:]

        senders = torch.repeat_interleave(
            torch.arange(num_positions, device=device), num_neighbors_per_position
        )
        receivers_imgs = index_array[within_radius]

        receivers = receivers_imgs % num_positions
        vectors = supercell_positions[receivers_imgs] - positions[senders]
        stacked = torch.stack((senders, receivers), dim=0)
        return stacked, vectors

    else:
        return positions_to_graph(supercell_positions, positions, radius, max_number_neighbors, library, n_workers)
