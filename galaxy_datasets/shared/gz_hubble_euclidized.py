import os
import logging

import pandas as pd

from zoobot.shared import label_metadata

from galaxy_datasets.shared import download_utils, label_metadata
from galaxy_datasets.check_internal_urls import INTERNAL_URLS_EXIST
if not INTERNAL_URLS_EXIST:
    raise FileNotFoundError
from galaxy_datasets.shared import internal_urls


def gz_hubble_euclidized(root, train, download):
    logging.info('Setting up gz_hubble_euclidized dataset')
    resources = [
        (internal_urls.hubble_euclidised_train_catalog, '99a92bb1abd47966f77a7b04e4e310ac'),  # train catalog
        (internal_urls.hubble_euclidised_test_catalog, '4f628f5e7d3282fbfa7ed4b0d7d91b11'),  # test catalog
        (internal_urls.hubble_euclidised_images, 'baee3a003631be238bb9cd6650cf03bf')
    ]
    images_to_spotcheck = ['20000011.jpg', '20164189.jpg', '203191082.jpg']

    downloader = download_utils.DatasetDownloader(root, resources, images_to_spotcheck)
    if download is True:
        downloader.download()

    label_cols = label_metadata.hubble_ortho_label_cols

    useful_columns = label_cols + ['filename']
    if train:
        train_catalog_loc = os.path.join(root, 'catalog_euclid_train.parquet')
        catalog = pd.read_parquet(train_catalog_loc, columns=useful_columns)
    else:
        test_catalog_loc = os.path.join(root, 'catalog_euclid_test.parquet')
        catalog = pd.read_parquet(test_catalog_loc, columns=useful_columns)

    catalog['file_loc'] = catalog.apply(lambda x: os.path.join(downloader.image_dir, x['filename']), axis=1)  # no subfolder

    logging.info('gz_hubble_euclidized dataset ready')
    return catalog, label_cols


if __name__ == '__main__':


    gz_hubble_euclidized(
        root='/nvme1/scratch/walml/repos/galaxy-datasets/roots/gz_hubble_euclidised',
        train=True,
        download=True
        )