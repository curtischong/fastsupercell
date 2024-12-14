from create_graph import compute_pbc_radius_graph
from prep_datasets import load_dataset


if __name__ == "__main__":
    dataset = load_dataset("datasets/alexandria_hdf5/train_10.h5")
    print(dataset)
    for config in dataset:
        frac_coord = config.frac_coord
        lattice = config.lattice
        atomic_numbers = config.atomic_numbers
        print(f"frac_coord: {frac_coord}")
        print(f"lattice: {lattice}")
        print(f"atomic_numbers: {atomic_numbers}")

        edge_index, displacements = compute_pbc_radius_graph(
            frac_coord,
            lattice,
            atomic_numbers,
            radius=1.0,
            max_number_neighbors=20,
            # brute_force=True
        )

        # assert the edge index is the same between the two approaches