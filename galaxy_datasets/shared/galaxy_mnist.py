import os
import logging

import pandas as pd

from galaxy_datasets.shared import download_utils


def galaxy_mnist(root, train, download):
    logging.info('Setting up galaxy_mnist dataset')
    resources = [
        ('https://dl.dropboxusercontent.com/s/6xlym0ney5q9aec14vvy1/galaxy_mnist_train_catalog.parquet?rlkey=fc5knqscwk2h6r4z5d3156vzr&dl=0', 'cd22b21d165802f4bc1adf997424aec2'),  # train catalog
        ('https://dl.dropboxusercontent.com/s/9assxy247i1nq8wjy7o0a/galaxy_mnist_test_catalog.parquet?rlkey=y1ns24xw0tcyntodi61rv14fu&dl=0', 'c5cca8d8afb6fb0d59baeb310a35d594'),  # test catalog
        ('https://dl.dropboxusercontent.com/s/lj89307kjx5plme9f66js/galaxy_mnist_images.tar.gz?rlkey=gvlwx2rl3hqpo3gplb9zlqrz0', '1c6cb0447f2f7ed676c3363ee194ced9')  # the images
    ]
    images_to_spotcheck = [
        'images/92abc34c-824c-4d46-a79c-cc28a768149f.jpg',
        'images/a4bd154c-327c-4064-90d2-7d3d2bd045dc.jpg'
    ]

    downloader = download_utils.DatasetDownloader(root, resources, images_to_spotcheck)
    if download is True:
        downloader.download()

    if train:
        train_catalog_loc = os.path.join(root, 'galaxy_mnist_train_catalog.parquet')
        catalog = pd.read_parquet(train_catalog_loc)
    else:
        test_catalog_loc = os.path.join(root, 'galaxy_mnist_test_catalog.parquet')
        catalog = pd.read_parquet(test_catalog_loc)

    catalog['file_loc'] = catalog.apply(lambda x: os.path.join(downloader.image_dir, x['filename']), axis=1)

    label_cols = ['label']  # 0-3, see https://github.com/mwalmsley/galaxy_mnist

    logging.info('galaxy_mnist dataset ready')
    return catalog, label_cols


if __name__ == '__main__':

    logging.basicConfig(level=logging.INFO)
    
    df, _ = galaxy_mnist(root='/home/walml/repos/galaxy-datasets/roots/galaxy_mnist', train=True, download=True)
    example_file = df['file_loc'][0]
    assert os.path.isfile(example_file), example_file
