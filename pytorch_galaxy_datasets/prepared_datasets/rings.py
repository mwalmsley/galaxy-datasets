import os

import pandas as pd
from pytorch_galaxy_datasets import galaxy_datamodule, galaxy_dataset, download_utils

from pytorch_galaxy_datasets.prepared_datasets import internal_urls

# TODO could eventually refactor this out of Zoobot as well
from zoobot.shared import label_metadata


class RingsDataset(galaxy_dataset.GalaxyDataset):
    
    def __init__(self, root, train=True, download=False, transform=None, target_transform=None):

        catalog, label_cols = rings_setup(root, train, download)

        super().__init__(catalog, label_cols, transform, target_transform)


def rings_setup(root, train, download):
    resources = [
        (internal_urls.rings_train_catalog, 'e2fb6b2bca45cd7f1c58f5b4089a5976'),  # train catalog
        (internal_urls.rings_test_catalog, '6e3f362a6e19ecd02675eaa48f6727f0'),  # test catalog
        (internal_urls.rings_images, 'd0950250436a05ce88de747e6af825b6')  # the images
    ]
    images_to_spotcheck = []

    downloader = download_utils.DatasetDownloader(root, resources, images_to_spotcheck)
    if download is True:
        downloader.download()

    # useful_columns = label_cols + ['subfolder', 'filename']
    useful_columns = None
    if train:
        train_catalog_loc = os.path.join(root, 'rings_train_catalog.parquet')
        catalog = pd.read_parquet(train_catalog_loc, columns=useful_columns)
    else:
        test_catalog_loc = os.path.join(root, 'rings_test_catalog.parquet')
        catalog = pd.read_parquet(test_catalog_loc, columns=useful_columns)

    catalog['file_loc'] = catalog.apply(lambda x: os.path.join(root, downloader.image_dir, x['subfolder'], x['filename']), axis=1)

    label_cols = label_metadata.rings_label_cols

    return catalog, label_cols


if __name__ == '__main__':

    # first download is basically just a convenient way to get the images and canonical catalogs
    rings_datset = RingsDataset(
        root='/nvme1/scratch/walml/repos/pytorch-galaxy-datasets/roots/decals_dr5',
        train=True,
        download=False
    )
    rings_catalog = rings_datset.catalog
    adjusted_catalog = rings_catalog.sample(1000)

    # user will probably tweak and use images/catalog directly for generic galaxy catalog datamodule
    # (which makes its own generic datasets internally)
    datamodule = galaxy_datamodule.GalaxyDataModule(
        label_cols=['ring_fraction'],  # counts and totals also available
        catalog=adjusted_catalog
    )

    datamodule.prepare_data()
    datamodule.setup()
    for images, labels in datamodule.train_dataloader():
        print(images.shape, labels.shape)
        break
        