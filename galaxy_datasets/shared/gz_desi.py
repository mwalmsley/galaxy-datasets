import os
import logging
from typing import Tuple, List

import pandas as pd

# TODO could eventually refactor this out of Zoobot as well
from galaxy_datasets.shared import label_metadata

from galaxy_datasets.shared import download_utils
from galaxy_datasets.check_internal_urls import INTERNAL_URLS_EXIST
if not INTERNAL_URLS_EXIST:
    raise FileNotFoundError
from galaxy_datasets.shared import internal_urls

"""
This downloads the *labelled* GZ DESI galaxies, at all redshifts, with no division between data releases (except label_cols)
useful to conveniently download galaxies for GZ DESI itself (see gz-decals-classifiers, though this also uses catalogs directly)
or for other large-scale supervised learning problems or for pretraining ahead of active learning
does not include unlabelled galaxies (for e.g. contrastive learning, active learning)

Added to make GZ DESI easily downloadable, basically
"""
def gz_desi(root, train, download) -> Tuple[pd.DataFrame, List]:
    logging.info('Setting up gz_desi dataset')
    resources = [
        (internal_urls.gz_desi_train_catalog, '50997398a10886dafbe556d071ffbc68'),
        (internal_urls.gz_desi_test_catalog, '386f13daf123f439ffb8266b65eac474'),
        (internal_urls.gz_desi_images_chunk_00, '1d52898581bfb8f08acd13fe77c69901'),
        (internal_urls.gz_desi_images_chunk_01, 'ada3fe9a94258075bbe62d42e6550440'),
        (internal_urls.gz_desi_images_chunk_02, 'e187ef2d77ea606244c8631599a31ca1'),
        (internal_urls.gz_desi_images_chunk_03, '1a649787bd2ecfe3efcb1856a3627548'),
        (internal_urls.gz_desi_images_chunk_04, 'a7734ef953f68aac7787b8ea9caf4633'),
        (internal_urls.gz_desi_images_chunk_05, '960b25c4f6fec9967625696bbfa72d07'),
        (internal_urls.gz_desi_images_chunk_06, 'bfa6fa0b64e49da72053b4f63468afa0'),
        (internal_urls.gz_desi_images_chunk_07, '1c1b59cc72ef86db765f5f7189c06260'),
        (internal_urls.gz_desi_images_chunk_08, 'abc32a75cb5eea4d425085abb6ecd6e8')
    ]
    images_to_spotcheck = []  # TODO

    downloader = download_utils.DatasetDownloader(root, resources, images_to_spotcheck, archive_includes_subdir=False)
    if download is True:
        downloader.download()


    # useful_columns = label_cols + ['subfolder', 'filename']
    if train:
        train_catalog_loc = os.path.join(root, 'gz_desi_train_catalog.parquet')
        catalog = pd.read_parquet(train_catalog_loc)
    else:
        test_catalog_loc = os.path.join(root, 'gz_desi_test_catalog.parquet')
        catalog = pd.read_parquet(test_catalog_loc)

    # removed 'root' from here as downloader.image_dir already includes root
    catalog['file_loc'] = catalog.apply(lambda x: os.path.join(downloader.image_dir, x['subfolder'], x['filename']), axis=1)

    # default, but not actually used here
    label_cols = label_metadata.decals_all_campaigns_ortho_label_cols
    logging.info('gz_desi dataset ready')
    return catalog, label_cols

