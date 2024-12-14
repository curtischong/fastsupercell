# can we use a linear transformation so instead of a parallelepiped, we create the graph in a euclidian plane?
# this way, when calculating the distances between two atoms, we can use a bitmask
# maybe diff axis will have diff "length" cause of the mapped distances
# now we can generate the edges VERY fast
# a future optimization would be to use sram instead

from create_graph import positions_to_graph


def fast(*, lattice, radius:int, max_number_neighbors: int, knn_library: str, n_workers: int = 1):

    return positions_to_graph(supercell_positions, positions, radius, max_number_neighbors, knn_library, n_workers)