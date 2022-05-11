import os

import pandas as pd
from pytorch_galaxy_datasets import galaxy_datamodule, galaxy_dataset, download_utils

# TODO could eventually refactor this out of Zoobot as well
from zoobot.shared import label_metadata


# TODO add train flag? currently full dataset
class GZ2Dataset(galaxy_dataset.GalaxyDataset):
    
    def __init__(self, root, train=True, download=False, transform=None, target_transform=None):

        catalog, label_cols = gz2_setup(root, train, download)  # no train arg

        super().__init__(catalog, label_cols, transform, target_transform)


def gz2_setup(root, train, download):
    resources = [
        ('https://dl.dropboxusercontent.com/s/vu77e3sh2s5c250/gz2_train_catalog.parquet', 'f489c9ec7dcf8d99f728bd00ee00b1d0'),  # the train catalog
        ('https://dl.dropboxusercontent.com/s/t8eh6f3oupndpl3/gz2_test_catalog.parquet', '8b2d74c93d35f86cc34f1d058b3b220b'),  # the test catalog
        ('https://zenodo.org/record/3565489/files/images_gz2.zip', 'bc647032d31e50c798770cf4430525c7')  # the images
    ]
    images_to_spotcheck = ['100097.jpg']

    downloader = download_utils.DatasetDownloader(root, resources, images_to_spotcheck)
    if download is True:
        downloader.download()


    # useful_columns = label_cols + ['filename']
    useful_columns = None  # quite a variety of possiblities - load all of them
    if train:
        train_catalog_loc = os.path.join(root, 'gz2_train_catalog.parquet')
        catalog = pd.read_parquet(train_catalog_loc, columns=useful_columns)
    else:
        test_catalog_loc = os.path.join(root, 'gz2_test_catalog.parquet')
        catalog = pd.read_parquet(test_catalog_loc, columns=useful_columns)

    catalog['file_loc'] = catalog['filename'].apply(
        lambda x: os.path.join(downloader.image_dir, x))

    label_cols = label_metadata.gz2_ortho_label_cols  # default, but you can ignore
    return catalog, label_cols


if __name__ == '__main__':

    # can use this directly, e.g. to visualise the dataset
    gz2_dataset = GZ2Dataset(
        root='/nvme1/scratch/walml/repos/pytorch-galaxy-datasets/roots/gz2',
        download=False
    )
    gz2_catalog = gz2_dataset.catalog  # and can get the catalog/cols for tweaking
    gz2_label_cols = gz2_dataset.label_cols

    # or, can use the setup method directly
    catalog, label_cols = gz2_setup(
        root='/nvme1/scratch/walml/repos/pytorch-galaxy-datasets/roots/gz2',
        download=False
    )
    adjusted_catalog = gz2_catalog.sample(1000)

    datamodule = galaxy_datamodule.GalaxyDataModule(
        label_cols=['smooth-or-featured_smooth'],
        catalog=adjusted_catalog
    )

    datamodule.prepare_data()
    datamodule.setup()
    for images, labels in datamodule.train_dataloader():
        print(images.shape, labels.shape)
        break