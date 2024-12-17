
from collections import Counter
import torch
from prep_datasets import load_dataset


if __name__ == "__main__":
    torch.set_printoptions(sci_mode=False)  # Disable scientific notation
    dataset = load_dataset("datasets/alexandria_hdf5/train_all.h5")

    atoms_cnt = Counter()
    for config in dataset:
        atoms_cnt[len(config.frac_coord)] += 1
    for k, v in sorted(atoms_cnt.items()):
        print(f"{k}: {v}")
