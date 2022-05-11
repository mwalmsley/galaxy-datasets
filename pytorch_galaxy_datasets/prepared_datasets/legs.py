import logging

import pandas as pd
from pytorch_galaxy_datasets import galaxy_datamodule, galaxy_dataset, download_utils
from pytorch_galaxy_datasets.prepared_datasets import internal_urls

# TODO could eventually refactor this out of Zoobot as well
from zoobot.shared import label_metadata


class LegsDataset(galaxy_dataset.GalaxyDataset):
    
    # based on https://pytorch.org/vision/stable/generated/torchvision.datasets.STL10.html
    def __init__(self, root=None, split='train', download=False, transform=None, target_transform=None, train=None):
        # train=None is just an exception-raising parameter to avoid confused users using the train=False api

        label_cols, catalog = legs_setup(root, split, download, train)

        # paths are not adjusted as cannot be downloaded
        # catalog = _temp_adjust_catalog_paths(catalog)
        # catalog = adjust_catalog_dtypes(catalog, label_cols)

        super().__init__(catalog, label_cols, transform, target_transform)


def legs_setup(root, split, download, train=None):

    if train is not None:
        raise ValueError("This dataset has unlabelled data: use split='train', 'test', 'unlabelled' or 'train+unlabelled' rather than train=False etc")

    assert split in ['train', 'test', 'unlabelled', 'train+unlabelled']

    if root is not None:
        'Legacy Survey cannot be downloaded - ignoring root {}'.format(root)
        # TODO update for non-manchester users with a manual copy?

    # resources = (
    #     internal_urls.legs_train_catalog, 'bae2906e337bd114af013d02f3782473',
    #     internal_urls.legs_test_catalog, '20919fe512ee8ce4d267790e519fcbf8',
    #     internal_urls.legs_unlabelled_catalog, 'fbf287990add34d2249f584325bc9dca'
    # )
    # downloader = download_utils.DatasetDownloader(root, resources, images_to_spotcheck=[])
    if download is True:
        raise NotImplementedError
        # logging.warning('Only downloading catalogs - images are too large to download')
        # downloader.download()

    label_cols = label_metadata.decals_all_campaigns_ortho_label_cols

    usecols = label_cols + ['file_loc']

    train_catalog_loc = '/share/nas2/walml/repos/gz-decals-classifiers/data/decals/presplit_catalogs/legs_all_campaigns_ortho_dr8_only_train_catalog.parquet'
    test_catalog_loc = '/share/nas2/walml/repos/gz-decals-classifiers/data/decals/presplit_catalogs/legs_all_campaigns_ortho_dr8_only_test_catalog.parquet'
    unlabelled_catalog_loc = '/share/nas2/walml/repos/gz-decals-classifiers/data/decals/presplit_catalogs/legs_all_campaigns_ortho_dr8_only_unlabelled_catalog.parquet'  # values in label_cols are all 0

    catalogs = []

    if 'train' in split:
        catalogs += pd.read_parquet(train_catalog_loc, usecols=usecols)

    if 'test' in split:  # test+unlabelled not supported, but could add if needed
        catalogs += pd.read_parquet(test_catalog_loc, usecols=usecols)

    if 'unlabelled' in split:
        catalogs += pd.read_parquet(unlabelled_catalog_loc, usecols=usecols)

    catalog = pd.concat(catalogs, axis=0)
    catalog = catalog.sample(len(catalog)).reset_index(drop=True)

    return label_cols, catalog




if __name__ == '__main__':


    # first download is basically just a convenient way to get the images and canonical catalogs
    legs_dataset = LegsDataset(
        root='whatever',
        train=True,
        download=False  # will fail except on galahad
    )
    legs_catalog = legs_dataset.catalog
    adjusted_catalog = legs_catalog.sample(1000)

    # user will probably tweak and use images/catalog directly for generic galaxy catalog datamodule
    # (which makes its own generic datasets internally)
    datamodule = galaxy_datamodule.GalaxyDataModule(
        label_cols=label_metadata.decals_all_campaigns_label_cols,
        catalog=adjusted_catalog
    )

    datamodule.prepare_data()
    datamodule.setup()
    for images, labels in datamodule.train_dataloader():
        print(images.shape, labels.shape)
        break
        
