from test import load_dataset
from visualization import  visualize_lattice

if __name__ == "__main__":
    dataset = load_dataset("datasets/alexandria_hdf5/train_10.h5")
    fig = visualize_lattice(dataset[0].lattice)
    fig.show()