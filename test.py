from prep_datasets import load_dataset


if __name__ == "__main__":
    dataset = load_dataset("datasets/alexandria_hdf5/train_10.h5")
    print(dataset)