import os

import pandas as pd

# TODO could eventually refactor this out of Zoobot as well
from zoobot.shared import label_metadata

from galaxy_datasets.shared import download_utils


def gz2(root, train, download):
    resources = [
        ('https://dl.dropboxusercontent.com/s/vu77e3sh2s5c250/gz2_train_catalog.parquet', 'f489c9ec7dcf8d99f728bd00ee00b1d0'),  # the train catalog
        ('https://dl.dropboxusercontent.com/s/t8eh6f3oupndpl3/gz2_test_catalog.parquet', '8b2d74c93d35f86cc34f1d058b3b220b'),  # the test catalog
        ('https://zenodo.org/record/3565489/files/images_gz2.zip', 'bc647032d31e50c798770cf4430525c7')  # the images
    ]
    images_to_spotcheck = ['100097.jpg']

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

    catalog['file_loc'] = catalog['filename'].apply(
        lambda x: os.path.join(downloader.image_dir, x))

    label_cols = label_metadata.gz2_ortho_label_cols  # default, but you can ignore
    return catalog, label_cols
