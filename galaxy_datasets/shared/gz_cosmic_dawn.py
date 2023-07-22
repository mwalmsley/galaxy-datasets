import os
import logging

import pandas as pd

from galaxy_datasets.shared import download_utils, label_metadata
from galaxy_datasets.check_internal_urls import INTERNAL_URLS_EXIST
if not INTERNAL_URLS_EXIST:
    raise FileNotFoundError
from galaxy_datasets.shared import internal_urls


def gz_cosmic_dawn(root, train, download):
    logging.info('Setting up gz_cosmic_dawn dataset')
    resources = [
        (internal_urls.cosmic_dawn_train_catalog, 'ed412bc070166858a5cc965b98204268'),  # train catalog
        (internal_urls.cosmic_dawn_test_catalog, '0c17fc4c7f2257c90c71edee6d95129a'),  # test catalog
        (internal_urls.cosmic_dawn_images, 'e95655312b48c23d1273369fbeb15884')  # the images
    ]
    images_to_spotcheck = [
        '929153cc-ee0f-4142-a37b-0bb33279585e.jpeg'
    ]

    downloader = download_utils.DatasetDownloader(root, resources, images_to_spotcheck)
    if download is True:
        downloader.download()

    # useful_columns = label_cols + ['subfolder', 'filename']
    useful_columns = None
    if train:
        train_catalog_loc = os.path.join(root, 'cosmic_dawn_train_catalog.parquet')
        catalog = pd.read_parquet(train_catalog_loc, columns=useful_columns)
    else:
        test_catalog_loc = os.path.join(root, 'cosmic_dawn_test_catalog.parquet')
        catalog = pd.read_parquet(test_catalog_loc, columns=useful_columns)

    catalog['file_loc'] = catalog.apply(lambda x: os.path.join(downloader.image_dir, x['filename']), axis=1)

    label_cols = label_metadata.rings_label_cols

    logging.info('gz_cosmic_dawn dataset ready')
    return catalog, label_cols


if __name__ == '__main__':

    logging.basicConfig(level=logging.INFO)
    
    gz_cosmic_dawn(root='/Users/user/repos/galaxy-datasets/roots/gz_cosmic_dawn', train=True, download=True)