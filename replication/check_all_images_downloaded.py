import os

from galaxy_datasets.pytorch.datasets import GZDesiDataset

if __name__ == '__main__':

    # or any other dataset
    dataset = GZDesiDataset(root='/nvme1/scratch/walml/repos/pytorch-galaxy-datasets/roots/gz_desi', train=True, download=False)
    for loc in dataset.catalog['file_loc']:
        if not os.path.isfile(loc):
            raise FileNotFoundError(loc)

    print('All images successfully located')