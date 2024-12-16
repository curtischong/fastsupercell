import torch
from fast_approach import extend_lattice
from test import load_dataset
from visualization import  plot_with_parallelopied, visualize_lattice

if __name__ == "__main__":
    dataset = load_dataset("datasets/alexandria_hdf5/train_10.h5")
    lattice = dataset[0].lattice
    extended_lattice, position_offset = extend_lattice(torch.Tensor(lattice), 3.0)
    fig = visualize_lattice(extended_lattice)
    plot_with_parallelopied(fig, lattice, color="#ff0000")
    fig.show()
    # inverse_lattice = np.linalg.inv(dataset[0].lattice)
    # # fig = visualize_lattice(inverse_lattice)
    # fig.show()