import os
import logging

import pandas as pd

from galaxy_datasets.shared import download_utils, label_metadata
from galaxy_datasets.check_internal_urls import INTERNAL_URLS_EXIST
if not INTERNAL_URLS_EXIST:
    raise FileNotFoundError
from galaxy_datasets.shared import internal_urls


def gz_euclid(root, train, download):
    logging.info('Setting up gz_euclid dataset')
    resources = [
        (internal_urls.gz_euclid_train_catalog, 'e65040c6dcc3e490634120e8fa249c74'),  # train catalog
        (internal_urls.gz_euclid_test_catalog, '1986f944ddbe47d2ae94beeca99ae414'),  # test catalog
        (internal_urls.gz_euclid_images, 'ba0fd28c170fa6e39c7bdd1d6cee202a')  # the images
    ]
    images_to_spotcheck = [
        'F-006_102011288_NEG842787454598311005.jpg',
        'F-006_102032117_NEG827217021378563764.jpg',
        'F-006_102042983_NEG918837531292461296.jpg'
    ]

    downloader = download_utils.DatasetDownloader(root, resources, images_to_spotcheck)
    if download is True:
        downloader.download()

    label_cols = label_metadata.euclid_ortho_label_cols
    useful_columns = label_cols + ['filename']
    # useful_columns = None
    if train:
        train_catalog_loc = os.path.join(root, 'gz_euclid_train_catalog.parquet')
        catalog = pd.read_parquet(train_catalog_loc, columns=useful_columns)
    else:
        test_catalog_loc = os.path.join(root, 'gz_euclid_test_catalog.parquet')
        catalog = pd.read_parquet(test_catalog_loc, columns=useful_columns)

    catalog['file_loc'] = catalog.apply(lambda x: os.path.join(downloader.image_dir, x['filename']), axis=1)

    logging.info('gz_euclid dataset ready')
    return catalog, label_cols


if __name__ == '__main__':

    logging.basicConfig(level=logging.INFO)
    
    gz_euclid(root='/home/walml/repos/galaxy-datasets/roots/gz_euclid_debug', train=True, download=True)
