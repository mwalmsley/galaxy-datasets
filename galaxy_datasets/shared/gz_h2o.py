import os
import logging

import pandas as pd

from galaxy_datasets.shared import download_utils, label_metadata
from galaxy_datasets.check_internal_urls import INTERNAL_URLS_EXIST
if not INTERNAL_URLS_EXIST:
    raise FileNotFoundError
from galaxy_datasets.shared import internal_urls


def gz_h2o(root, train, download):
    logging.info('Setting up gz_h2o dataset')
    resources = [
        (internal_urls.gz_h2o_train_catalog, '4fa5003cdc1c3beb137be69c06bf796c'),  # train catalog
        (internal_urls.gz_h2o_test_catalog, 'ab4f58091f1c0dee01d9c884c4e9311e'),  # test catalog
        (internal_urls.gz_h2o_images, '805ef596b4271288607ddda5a6acf60a')  # the images
    ]
    images_to_spotcheck = [
        'lz_595/lz_59532_cutout.jpg'
    ]

    downloader = download_utils.DatasetDownloader(root, resources, images_to_spotcheck)
    if download is True:
        downloader.download()

    label_cols = label_metadata.cosmic_dawn_ortho_label_cols
    useful_columns = label_cols + ['filename', 'subfolder', 'ra', 'dec']
    # useful_columns = None
    if train:
        train_catalog_loc = os.path.join(root, 'h2o_train_catalog.parquet')
        catalog = pd.read_parquet(train_catalog_loc, columns=useful_columns)
    else:
        test_catalog_loc = os.path.join(root, 'h2o_test_catalog.parquet')
        catalog = pd.read_parquet(test_catalog_loc, columns=useful_columns)

    catalog['file_loc'] = catalog.apply(lambda x: os.path.join(downloader.image_dir, x['subfolder'], x['filename']), axis=1)

    logging.info('gz_h2o dataset ready')
    return catalog, label_cols


if __name__ == '__main__':

    logging.basicConfig(level=logging.INFO)
    
    gz_h2o(root='/home/walml/repos/galaxy-datasets/roots/gz_h2o_debug', train=True, download=True)
