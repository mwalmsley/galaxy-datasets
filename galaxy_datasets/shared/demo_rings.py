import os
import logging

import pandas as pd

from galaxy_datasets.shared import download_utils


def demo_rings(root, train, download):
    logging.info('Setting up demo_rings dataset')
    resources = [
        ('https://dl.dropboxusercontent.com/s/4cfb9xbnlkvbkv1/demo_rings_train_catalog.parquet', '6b3ff71b8893175cbdb9dc092ed0dbd9'),  # train catalog
        ('https://dl.dropboxusercontent.com/s/xghqujm41ujjlko/demo_rings_test_catalog.parquet', 'e63e13b57e8dbe55c1cf339dee5395ae'),  # test catalog
        ('https://dl.dropboxusercontent.com/s/80s5j6dygtw0gu6/demo_rings_images.tar.gz', 'b8489af55d54b6965d4b037a4f952f3c')  # the images
    ]
    images_to_spotcheck = [
        'images/313982/313982_3559.jpg',
        'images/374032/374032_255.jpg'
    ]

    downloader = download_utils.DatasetDownloader(root, resources, images_to_spotcheck)
    if download is True:
        downloader.download()

    if train:
        train_catalog_loc = os.path.join(root, 'demo_rings_train_catalog.parquet')
        catalog = pd.read_parquet(train_catalog_loc)
    else:
        test_catalog_loc = os.path.join(root, 'demo_rings_test_catalog.parquet')
        catalog = pd.read_parquet(test_catalog_loc)

    catalog['file_loc'] = catalog.apply(lambda x: os.path.join(downloader.image_dir, x['filename']), axis=1)

    label_cols = ['ring']  # 1 or 0

    logging.info('demo_rings dataset ready')
    return catalog, label_cols


if __name__ == '__main__':

    logging.basicConfig(level=logging.INFO)
    
    demo_rings(root='/Users/user/repos/galaxy-datasets/roots/demo_rings', train=True, download=True)