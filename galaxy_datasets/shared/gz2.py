import os
import logging

import pandas as pd

from galaxy_datasets.shared import download_utils, label_metadata


def gz2(root, train, download):
    logging.info('Setting up gz2 dataset')
    resources = [
        ('https://dl.dropboxusercontent.com/s/vu77e3sh2s5c250/gz2_train_catalog.parquet', 'd5507a9332c75fd84a7f2562567da36b'),  # the train catalog
        ('https://dl.dropboxusercontent.com/s/t8eh6f3oupndpl3/gz2_test_catalog.parquet', 'ac1fba88d0e8a95ee4f4eef79ea03063'),  # the test catalog
        ('https://dl.dropboxusercontent.com/s/5so7yof2afe761p/images_gz2.tar.gz', 'e3eab2fec57a6a60577236b9e0a6913d')  # the images
    ]
    images_to_spotcheck = ['587722/587722981741363294.jpg']

    downloader = download_utils.DatasetDownloader(root, resources, images_to_spotcheck)
    if download is True:
        downloader.download()


    # useful_columns = label_cols + ['filename']
    useful_columns = None  # quite a variety of possiblities - load all of them
    if train:
        train_catalog_loc = os.path.join(root, 'gz2_train_catalog.parquet')
        catalog = pd.read_parquet(train_catalog_loc, columns=useful_columns)
    else:
        test_catalog_loc = os.path.join(root, 'gz2_test_catalog.parquet')
        catalog = pd.read_parquet(test_catalog_loc, columns=useful_columns)

    catalog['file_loc'] = catalog.apply(
        lambda x: os.path.join(downloader.image_dir, x['subfolder'], x['filename']),
        axis=1
    )

    label_cols = label_metadata.gz2_ortho_label_cols  # default, but you can ignore
    logging.info('gz2 dataset ready')
    return catalog, label_cols

