import os
import logging

import pandas as pd

from galaxy_datasets.shared import download_utils


def demo_gz_candels(root, train, download):
    logging.info('Setting up demo_gz_candels dataset')
    resources = [
        ('https://dl.dropboxusercontent.com/s/rdejdbzn3vsid3r/demo_gz_candels_train_catalog.parquet', '586632f9f832748dec2a52a5375bf49e'),  # train catalog
        ('https://dl.dropboxusercontent.com/s/2dn018co3ip7f8t/demo_gz_candels_test_catalog.parquet',  '2519671bfac59a84d971402d0ab8094c'), # test catalog
        ('https://dl.dropboxusercontent.com/s/1nhhvxsi53rjx7a/demo_gz_candels_images.tar.gz', 'fc2d856cc8d780b381b44681cadf57d6')  # the images
    ]
    images_to_spotcheck = [
        'images/COS_27104.jpg',
        'images/UDS_13502.jpg'
    ]

    downloader = download_utils.DatasetDownloader(root, resources, images_to_spotcheck)
    if download is True:
        downloader.download()

    if train:
        train_catalog_loc = os.path.join(root, 'demo_gz_candels_train_catalog.parquet')
        catalog = pd.read_parquet(train_catalog_loc)
    else:
        test_catalog_loc = os.path.join(root, 'demo_gz_candels_test_catalog.parquet')
        catalog = pd.read_parquet(test_catalog_loc)

    catalog['file_loc'] = catalog.apply(lambda x: os.path.join(downloader.image_dir, x['filename']), axis=1)

    label_cols = ['ring']  # 1 or 0

    logging.info('demo_gz_candels dataset ready')
    return catalog, label_cols


if __name__ == '__main__':

    logging.basicConfig(level=logging.INFO)
    
    demo_gz_candels(root='/Users/user/repos/galaxy-datasets/roots/demo_gz_candels', train=True, download=True)