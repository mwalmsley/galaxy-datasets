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
        (internal_urls.gz_euclid_train_catalog, '350a71f59504b0a1b1400a6396d24531'),  # train catalog
        (internal_urls.gz_euclid_test_catalog, 'd43c2b0ee74dd9a31a738f0929927078'),  # test catalog
        (internal_urls.gz_euclid_images, '84d7db3cf607e2c36fd35890f693b8bd')  # the images
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
