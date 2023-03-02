import os
import logging

import pandas as pd

from galaxy_datasets.shared import download_utils, label_metadata
from galaxy_datasets.check_internal_urls import INTERNAL_URLS_EXIST
if not INTERNAL_URLS_EXIST:
    raise FileNotFoundError
from galaxy_datasets.shared import internal_urls


def gz_rings(root, train, download):
    logging.info('Setting up gz_rings dataset')
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

    catalog['file_loc'] = catalog.apply(lambda x: os.path.join(downloader.image_dir, x['subfolder'], x['filename']), axis=1)

    label_cols = label_metadata.rings_label_cols

    logging.info('gz_rings dataset ready')
    return catalog, label_cols


if __name__ == '__main__':

    logging.basicConfig(level=logging.INFO)
    
    gz_rings(root='/Users/user/repos/galaxy-datasets/roots/gz_rings', train=True, download=True)