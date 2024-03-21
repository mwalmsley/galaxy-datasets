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
        (internal_urls.gz_desi_train_catalog, 'edcbbe4f0adb767d26bdc5de26aa854d'),
        (internal_urls.gz_desi_test_catalog, '95bfed06c5a00742f58171f46cb1d8c1'),
        (internal_urls.gz_desi_images_chunk_00, '76c861762633d022603686e0794ce99c'),
        (internal_urls.gz_desi_images_chunk_01, '844502e49d6e087b6c1f4fcfef0772c0'),
        (internal_urls.gz_desi_images_chunk_02, 'b793771b8fc9383b616d2dc67bdfbf7e'),
        (internal_urls.gz_desi_images_chunk_03, '003821161b4990a49d1dcb5d72d49b12'),
        (internal_urls.gz_desi_images_chunk_04, 'aa85dec158d95edea98b31efe30231d5'),
        (internal_urls.gz_desi_images_chunk_05, '36ed5893441a533d1d74fe480cb42cb2'),
        (internal_urls.gz_desi_images_chunk_06, '6b84e92c7b4f1edbb372d2247ff1740a'),
        (internal_urls.gz_desi_images_chunk_07, 'feef448b45eeb3d5edc3f44d936f386a'),
        (internal_urls.gz_desi_images_chunk_08, 'c8be005044f94046a7080617815a3d00')
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

