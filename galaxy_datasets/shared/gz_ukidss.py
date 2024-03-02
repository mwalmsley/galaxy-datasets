import os
import logging

import pandas as pd

from galaxy_datasets.shared import download_utils, label_metadata
from galaxy_datasets.check_internal_urls import INTERNAL_URLS_EXIST
if not INTERNAL_URLS_EXIST:
    raise FileNotFoundError
from galaxy_datasets.shared import internal_urls


def gz_ukidss(root, train, download):
    logging.info('Setting up gz_ukidss dataset')
    resources = [
        (internal_urls.gz_ukidss_train_catalog, '8d84de1a1f33556f83470ce0c22c979b'),  # train catalog
        (internal_urls.gz_ukidss_test_catalog, 'f64b850e941b09546df5aca84b708fb4'),  # test catalog
        (internal_urls.gz_ukidss_images, 'd4586af013b1543753346544b4703711')  # the images
    ]
    images_to_spotcheck = [
        '524482bc3ae74054bf009/524482bc3ae74054bf009a00.jpg'
    ]

    downloader = download_utils.DatasetDownloader(root, resources, images_to_spotcheck)
    if download is True:
        downloader.download()

    label_cols = label_metadata.ukidss_ortho_label_cols
    useful_columns = label_cols + ['filename', 'subfolder']
    # useful_columns = None
    if train:
        train_catalog_loc = os.path.join(root, 'ukidss_train_catalog.parquet')
        catalog = pd.read_parquet(train_catalog_loc, columns=useful_columns)
    else:
        test_catalog_loc = os.path.join(root, 'ukidss_test_catalog.parquet')
        catalog = pd.read_parquet(test_catalog_loc, columns=useful_columns)

    catalog['file_loc'] = catalog.apply(lambda x: os.path.join(downloader.image_dir, x['subfolder'], x['filename']), axis=1)

    logging.info('gz_ukidss dataset ready')
    return catalog, label_cols


if __name__ == '__main__':

    logging.basicConfig(level=logging.INFO)
    
    gz_ukidss(root='/home/walml/repos/galaxy-datasets/roots/gz_ukidss_debug', train=True, download=True)
