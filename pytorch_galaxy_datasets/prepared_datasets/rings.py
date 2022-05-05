import os

import pandas as pd
from pytorch_galaxy_datasets import galaxy_datamodule, galaxy_dataset, download_utils

from pytorch_galaxy_datasets.prepared_datasets import internal_urls

# TODO could eventually refactor this out of Zoobot as well
from zoobot.shared import label_metadata


class RingsDataset(galaxy_dataset.GalaxyDataset):
    
    def __init__(self, root, train=True, download=False, transform=None, target_transform=None):

        label_cols, catalog = rings_setup(root, train, download)

        super().__init__(catalog, label_cols, transform, target_transform)


def rings_setup(root, train, download):
    resources = [
        (internal_urls.rings_train_catalog, 'fdc96a200189d64085edd5e191cbd683'),  # train catalog
        (internal_urls.rings_test_catalog, '8929dea0c25d8e8afbd3870e50e9dca8'),  # test catalog
        (internal_urls.rings_images, '4dad52b1f9819059eb9793eef8cb320c')  # the images
    ]
    images_to_spotcheck = []

    downloader = download_utils.DatasetDownloader(root, resources, images_to_spotcheck)
    if download is True:
        downloader.download()

    label_cols = label_metadata.rings_label_cols

    useful_columns = label_cols + ['subfolder', 'filename']
    if train:
        train_catalog_loc = os.path.join(root, 'rings_train_catalog.parquet')
        catalog = pd.read_parquet(train_catalog_loc, columns=useful_columns)
    else:
        test_catalog_loc = os.path.join(root, 'rings_test_catalog.parquet')
        catalog = pd.read_parquet(test_catalog_loc, columns=useful_columns)

    catalog['file_loc'] = catalog.apply(lambda x: os.path.join(root, downloader.image_dir, x['subfolder'], x['filename']), axis=1)

    catalog = _temp_adjust_catalog_dtypes(catalog, label_cols)
    return label_cols,catalog



def _temp_adjust_catalog_dtypes(catalog, label_cols):
    # enforce datatypes
    for answer_col in label_cols:
        catalog[answer_col] = catalog[answer_col].astype(int)
    return catalog


if __name__ == '__main__':

    # first download is basically just a convenient way to get the images and canonical catalogs
    rings_datset = RingsDataset(
        root='/nvme1/scratch/walml/repos/pytorch-galaxy-datasets/tests/dr5_root',
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
        