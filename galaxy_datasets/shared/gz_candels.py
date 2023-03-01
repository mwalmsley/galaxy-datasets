
import os
import logging

import pandas as pd

from galaxy_datasets.shared import download_utils, label_metadata


def gz_candels(root, train, download):
    logging.info('Setting up gz_candels dataset')
    resources = [
        ('https://dl.dropboxusercontent.com/s/cnjvdinnhh1r1md/candels_ortho_train_catalog.parquet', '90593d1bab79a608cf0e645d6fd8e741'),  # train catalog
        ('https://dl.dropboxusercontent.com/s/y83v1gktw72hs0f/candels_ortho_test_catalog.parquet', '1062993dd8df09684b335678ab3fa8e3'),  # test catalog
        ('https://dl.dropboxusercontent.com/s/d67we9xsn8vyr5k/candels_images.tar.gz', 'b621ee4e650cf084a1a0c1fe5c9d0a21')  # the images
    ]
    images_to_spotcheck = ['COS_9933.jpg', 'UDS_21986.jpg', 'GDS_9405.jpg']

    downloader = download_utils.DatasetDownloader(root, resources, images_to_spotcheck)
    if download is True:
        downloader.download()

    # label_cols = [question + answer for question, answers in candels_ortho_pairs.items() for answer in answers]  # defined below, globally in this script (for imports elsewhere)
    label_cols = label_metadata.candels_ortho_label_cols  # see below

    useful_columns = label_cols + ['filename']
    if train:
        train_catalog_loc = os.path.join(root, 'candels_ortho_train_catalog.parquet')
        catalog = pd.read_parquet(train_catalog_loc, columns=useful_columns)
    else:
        test_catalog_loc = os.path.join(root, 'candels_ortho_test_catalog.parquet')
        catalog = pd.read_parquet(test_catalog_loc, columns=useful_columns)

    catalog['file_loc'] = catalog.apply(lambda x: os.path.join(downloader.image_dir, x['filename']), axis=1)  # no subfolder

    logging.info('gz_candels dataset ready')
    return catalog, label_cols


if __name__ == '__main__':

    gz_candels(root='roots/gz_candels', train=True, download=True)