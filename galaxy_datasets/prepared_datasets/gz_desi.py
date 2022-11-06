import os

import pandas as pd

# TODO could eventually refactor this out of Zoobot as well
from zoobot.shared import label_metadata

from galaxy_datasets.prepared_datasets import download_utils, internal_urls

"""
This downloads the *labelled* GZ DESI galaxies, at all redshifts, with no division between data releases (except label_cols)
useful to conveniently download galaxies for GZ DESI itself (see gz-decals-classifiers, though this also uses catalogs directly)
or for other large-scale supervised learning problems or for pretraining ahead of active learning
does not include unlabelled galaxies (for e.g. contrastive learning, active learning)

Added to make GZ DESI easily downloadable, basically
"""

def setup(root, train, download):
    resources = [
        (internal_urls.gz_desi_train_catalog, '64dcf77d1a67808e75eac32a8018f345'),
        (internal_urls.gz_desi_test_catalog, 'f3b554c4572e2fbba106d662b56523f1'),
        (internal_urls.gz_desi_images_chunk_00, ''),
        (internal_urls.gz_desi_images_chunk_01, ''),
        (internal_urls.gz_desi_images_chunk_02, ''),
        (internal_urls.gz_desi_images_chunk_03, ''),
        (internal_urls.gz_desi_images_chunk_04, ''),
        (internal_urls.gz_desi_images_chunk_05, ''),
        (internal_urls.gz_desi_images_chunk_06, ''),
        (internal_urls.gz_desi_images_chunk_07, ''),
        (internal_urls.gz_desi_images_chunk_08, '')
    ]
    images_to_spotcheck = []

    downloader = download_utils.DatasetDownloader(root, resources, images_to_spotcheck)
    if download is True:
        downloader.download()


    # useful_columns = label_cols + ['subfolder', 'filename']
    if train:
        train_catalog_loc = os.path.join(root, 'decals_dr5_ortho_train_catalog.parquet')
        catalog = pd.read_parquet(train_catalog_loc)
    else:
        test_catalog_loc = os.path.join(root, 'decals_dr5_ortho_test_catalog.parquet')
        catalog = pd.read_parquet(test_catalog_loc)

    # removed 'root' from here as downloader.image_dir already includes root
    catalog['file_loc'] = catalog.apply(lambda x: os.path.join(downloader.image_dir, x['subfolder'], x['filename']), axis=1)

    # default, but not actually used here
    label_cols = label_metadata.decals_dr5_ortho_label_cols
    return catalog, label_cols

