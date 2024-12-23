import torch
from create_graph import _compute_img_positions_torch, points_in_parallelepiped
from pruning_algo import create_masking_parallelepiped
from pruning_algo_incorrect import extend_lattice
from test import load_dataset
from visualization import  plot_points, plot_with_parallelepiped, visualize_lattice


def vis_scaled_lattice():
    dataset = load_dataset("datasets/alexandria_hdf5/train_10.h5")
    lattice = dataset[0].lattice
    extended_lattice, position_offset = extend_lattice(torch.Tensor(lattice), 3.0)
    fig = visualize_lattice(extended_lattice) # show the extended lattice first since it's larger and will scale the fig properly
    plot_with_parallelepiped(fig, lattice, color="#ff0000")
    fig.show()

def vis_points_masked_by_scaled_lattice():
    dataset = load_dataset("datasets/alexandria_hdf5/train_10.h5")
    material = dataset[0]
    lattice = material.lattice
    lattice_torch = torch.from_numpy(lattice)
    frac_coord = material.frac_coord
    cart_coord = frac_coord @ lattice

    radius = 1.1 # at this radius, there are points that are OUTSIDE the allowed parallelepiped area: e.g. one atom is 3.906957126077915 away from the origin, and is OUTSIDE the parallelepiped

    cart_supercell_coords = _compute_img_positions_torch(torch.from_numpy(cart_coord), lattice_torch)
    cart_supercell_coords = cart_supercell_coords.reshape(-1, 3)
    # cart_supercell_coords = supercell_coords @ lattice

    # extended_lattice1, position_offset1 = extend_lattice(lattice_torch, radius)
    extended_lattice, position_offset = create_masking_parallelepiped(lattice_torch, radius)

    # 1 plot the parallelepipeds
    fig = visualize_lattice(extended_lattice, position_offset.numpy()) # show the extended lattice first since it's larger and will scale the fig properly
    plot_with_parallelepiped(fig, lattice, color="#ff0000")
    # plot_with_parallelepiped(fig, extended_lattice1, translation=position_offset1.numpy(), color="#00ff00")

    # 2 plot the points
    plot_points(fig, cart_supercell_coords)

    # 3 plot the points in the lattice2
    lattice2_mask = points_in_parallelepiped(extended_lattice, position_offset, cart_supercell_coords)
    supercell2_positions = torch.masked_select(cart_supercell_coords, lattice2_mask.unsqueeze(-1))
    plot_points(fig, supercell2_positions.reshape(-1, 3), color="#0000ff") # original coords in original lattice

    # 4 plot the original points in the lattice
    plot_points(fig, cart_coord, color="#ff0000") # original coords in original lattice
    fig.show()

if __name__ == "__main__":
    # vis_scaled_lattice()
    vis_points_masked_by_scaled_lattice()