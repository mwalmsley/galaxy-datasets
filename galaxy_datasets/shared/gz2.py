import os
import logging

import pandas as pd

# TODO could eventually refactor this out of Zoobot as well
from zoobot.shared import label_metadata

from galaxy_datasets.shared import download_utils


def gz2(root, train, download):
    logging.info('Setting up gz2 dataset')
    resources = [
        ('https://dl.dropboxusercontent.com/s/vu77e3sh2s5c250/gz2_train_catalog.parquet', '4601cf8f7bea8ab0d79b2aada1a4663d'),  # the train catalog
        ('https://dl.dropboxusercontent.com/s/t8eh6f3oupndpl3/gz2_test_catalog.parquet', '65bc4d26cb1dd1426fed4f3c357cb42c'),  # the test catalog
        ('https://zenodo.org/record/3565489/files/images_gz2.zip', '4c085a3d2a30915f4daef365175f509d')  # the images
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
        axis=0
    )

    label_cols = label_metadata.gz2_ortho_label_cols  # default, but you can ignore
    logging.info('gz2 dataset ready')
    return catalog, label_cols
