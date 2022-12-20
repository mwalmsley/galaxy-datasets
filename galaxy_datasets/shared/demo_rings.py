import os
import logging

import pandas as pd

from galaxy_datasets.shared import download_utils


def demo_rings(root, train, download):
    logging.info('Setting up demo_rings dataset')
    resources = [
        ('https://dl.dropboxusercontent.com/s/qwf4irf0xzj96sm/demo_rings_train_catalog.parquet', 'cf650961bcdd889f201ed049fb825085'),  # train catalog
        ('https://dl.dropboxusercontent.com/s/b0w3ffoww4t9bq0/demo_rings_test_catalog.parquet', 'e84d19d279a02fb3a319d4e8aff4d724'),  # test catalog
        ('https://dl.dropboxusercontent.com/s/54016cw50ide3y8/demo_rings_images.tar.xz', '36a75712ce1929b0c376ba3a8d51b7a8')  # the images
    ]
    images_to_spotcheck = [
        'images/J000411.19+020924.0.jpg',
        'images/J235601.89+073123.0.jpg'
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
    
    demo_rings(root='/home/walml/repos/galaxy-datasets/roots/demo_rings', train=True, download=True)