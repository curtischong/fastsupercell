# can we use a linear transformation so instead of a parallelepiped, we create the graph in a euclidian plane?
# this way, when calculating the distances between two atoms, we can use a bitmask
# maybe diff axis will have diff "length" cause of the mapped distances
# now we can generate the edges VERY fast
# a future optimization would be to use sram instead

import numpy as np
from create_graph import _compute_img_positions_torch, positions_to_graph

# we transform all the points to a normal 1x1x1 cube
# then we just find the points near the edges of that cube
# lastly we map those points to the hypercube and do the kd tree calculation for edges


# https://shad.io/MatVis/

def fast(*, lattice, frac_coord, radius: int, max_number_neighbors: int, knn_library: str, n_workers: int = 1):
    cart_coords = np.dot(frac_coord, lattice)
    for side in [(1, 0, 0), (0, 1, 0), (0, 0, 1)], (-1, 0, 0), (0, -1, 0), (0, 0, -1):
        # radius_for_side
        supercell_positions = _compute_img_positions_torch(
            positions=cart_coords, periodic_boundaries=lattice
        )

    # we have all the coords. we just need to get the coords within the radius
    # is there a fast bitwise way to just get the coords within the radius?
    # it went to the norm of 

    return positions_to_graph(supercell_positions, positions=cart_coords, radius=radius, max_number_neighbors=max_number_neighbors, library=knn_library, n_workers=n_workers)