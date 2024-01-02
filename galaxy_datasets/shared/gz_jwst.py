import os
import logging

import pandas as pd

from galaxy_datasets.shared import download_utils, label_metadata
from galaxy_datasets.check_internal_urls import INTERNAL_URLS_EXIST
if not INTERNAL_URLS_EXIST:
    raise FileNotFoundError
from galaxy_datasets.shared import internal_urls


def gz_jwst(root, train, download):
    logging.info('Setting up gz_jwst dataset')
    resources = [
        (internal_urls.gz_jwst_train_catalog, '69e689017e8317abce6b2715382746ae'),  # train catalog
        (internal_urls.gz_jwst_test_catalog, 'c23807b06708bbdc5fdaefa284a96094'),  # test catalog
        (internal_urls.gz_jwst_images, '28ed102a4bcf4bb6693caeebd76a9c44')  # the images
    ]
    images_to_spotcheck = [
        '88113560.jpg'
    ]

    downloader = download_utils.DatasetDownloader(root, resources, images_to_spotcheck)
    if download is True:
        downloader.download()

    label_cols = label_metadata.jwst_ortho_label_cols
    useful_columns = label_cols + ['filename']
    # useful_columns = None
    if train:
        train_catalog_loc = os.path.join(root, 'jwst_train_catalog.parquet')
        catalog = pd.read_parquet(train_catalog_loc, columns=useful_columns)
    else:
        test_catalog_loc = os.path.join(root, 'jwst_test_catalog.parquet')
        catalog = pd.read_parquet(test_catalog_loc, columns=useful_columns)

    catalog['file_loc'] = catalog.apply(lambda x: os.path.join(downloader.image_dir, x['filename']), axis=1)

    logging.info('gz_jwst dataset ready')
    return catalog, label_cols


if __name__ == '__main__':

    logging.basicConfig(level=logging.INFO)
    
    gz_jwst(root='/home/walml/repos/galaxy-datasets/roots/gz_jwst_debug', train=True, download=False)
