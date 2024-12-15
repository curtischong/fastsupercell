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


def fast(*, lattice, frac_coord, radius: int = 5, max_number_neighbors: int, knn_library: str, n_workers: int = 1):
    cart_coords = np.dot(frac_coord, lattice)

    for normal in normal_vectors(lattice):
        # radius_for_side
        supercell_positions = _compute_img_positions_torch(
            positions=cart_coords, periodic_boundaries=lattice
        )
        normal_vector = ""
        # how does the normal vector change AFTER we inverse the lattice?
        # we can take a normal vector and run the reverse transformation
        # the radius for the side is the NORM of the reversed vector

    # we have all the coords. we just need to get the coords within the radius
    # is there a fast bitwise way to just get the coords within the radius?
    # it went to the norm of 

    return positions_to_graph(supercell_positions, positions=cart_coords, radius=radius, max_number_neighbors=max_number_neighbors, library=knn_library, n_workers=n_workers)

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