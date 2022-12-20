import os
import logging

import pandas as pd

from galaxy_datasets.shared import download_utils, label_metadata


def gz_hubble(root, train, download):
    logging.info('Setting up gz_hubble dataset')
    resources = [
        ('https://dl.dropboxusercontent.com/s/xnktj9hq6xig0a7/hubble_ortho_train_catalog.parquet', 'c6cb821f7ebefb583dc74488cf7bfc5f'),  # train catalog
        ('https://dl.dropboxusercontent.com/s/1g9lwih9944sys8/hubble_ortho_test_catalog.parquet', '05e01ed822b34400f32977280eebec87'),  # test catalog
        ('https://dl.dropboxusercontent.com/s/k9xco1mtp8bw60v/hubble_images.tar.gz', None)  # the images 'b6fed2463bb2d17ddb8302f6b060534a'
    ]
    images_to_spotcheck = ['10000325.jpg', '20163083.jpg']

    downloader = download_utils.DatasetDownloader(root, resources, images_to_spotcheck)
    if download is True:
        downloader.download()

    # label_cols = [question + answer for question, answers in hubble_ortho_pairs.items() for answer in answers]  # defined below, globally in this script (for imports elsewhere)
    label_cols = label_metadata.hubble_ortho_label_cols  # see below

    useful_columns = label_cols + ['filename']
    if train:
        train_catalog_loc = os.path.join(root, 'hubble_ortho_train_catalog.parquet')
        catalog = pd.read_parquet(train_catalog_loc, columns=useful_columns)
    else:
        test_catalog_loc = os.path.join(root, 'hubble_ortho_test_catalog.parquet')
        catalog = pd.read_parquet(test_catalog_loc, columns=useful_columns)

    catalog['file_loc'] = catalog.apply(lambda x: os.path.join(downloader.image_dir, x['filename']), axis=1)  # no subfolder

    logging.info('gz_hubble dataset ready')
    return catalog, label_cols


if __name__ == '__main__':

    hubble_cols = [x.replace('-hubble', '') for x in label_metadata.hubble_ortho_label_cols]
    # print('\n'.join(cols))

    decals_cols = [x for x in label_metadata.decals_label_cols]
    # print('\n'.join(cols))

    all_cols = list(set(hubble_cols).union(decals_cols))
    all_cols.sort()
    print('\n'.join(all_cols))
