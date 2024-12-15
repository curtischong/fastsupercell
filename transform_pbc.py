from test import load_dataset
from visualization import  visualize_lattice
import numpy as np

if __name__ == "__main__":
    dataset = load_dataset("datasets/alexandria_hdf5/train_10.h5")
    fig = visualize_lattice(dataset[0].lattice)
    fig.show()
    inverse_lattice = np.linalg.inv(dataset[0].lattice)
    # fig = visualize_lattice(inverse_lattice)
    fig.show()