import os
import logging

import pandas as pd

# this one has label_metadata contained in this repo, not zoobot. Others will follow, perhaps.
from zoobot.shared import label_metadata

from galaxy_datasets.shared import download_utils


def gz_hubble_euclidized(root, train, download):
    logging.info('Setting up gz_hubble_euclidized dataset')
    resources = [
        ('ADD TRAIN LABELS LINK HERE', 'HASH_TRAIN'),  # train catalog
        ('ADD TEST LABELS LINK HERE', 'HASH_TEST'),  # test catalog
        ('ADD IMAGES LINK HERE', None)
    ]
    images_to_spotcheck = ['20163083.jpeg']

    downloader = download_utils.DatasetDownloader(root, resources, images_to_spotcheck)
    if download is True:
        downloader.download()

    # label_cols = [question + answer for question, answers in hubble_ortho_pairs.items() for answer in answers]  # defined below, globally in this script (for imports elsewhere)
    label_cols = hubble_ortho_label_cols  # see below

    useful_columns = label_cols + ['filename']
    if train:
        train_catalog_loc = os.path.join(root, 'hubble_ortho_train_catalog.parquet')
        catalog = pd.read_parquet(train_catalog_loc, columns=useful_columns)
    else:
        test_catalog_loc = os.path.join(root, 'hubble_ortho_test_catalog.parquet')
        catalog = pd.read_parquet(test_catalog_loc, columns=useful_columns)

    catalog['file_loc'] = catalog.apply(lambda x: os.path.join(downloader.image_dir, x['filename']), axis=1)  # no subfolder

    logging.info('gz_hubble_euclidized dataset ready')
    return catalog, label_cols

# TODO may change features to featured-or-disk
hubble_pairs = {
    'smooth-or-featured': ['_smooth', '_features', '_artifact'],
    'how-rounded': ['_completely', '_in-between', '_cigar-shaped'],
    'clumpy-appearance': ['_yes', '_no'],
    'clump-count': ['_1', '_2', '_3', '_4', '_5-plus', '_cant-tell'],
    # disable these for now as I don't support having several but not all answers leading to the same next question
    # 'clump-configuration': ['_straight-line', '_chain', '_cluster-or-irregular', '_spiral'],
    # 'one-clump-brightest': ['_yes', '_no'],
    # 'brightest-clump-central': ['_yes', '_no'],
    'disk-edge-on': ['_yes', '_no'],
    'bulge-shape': ['_rounded', '_boxy', '_none'],
    'bar': ['_yes', '_no'],
    'has-spiral-arms': ['_yes', '_no'],
    'spiral-winding': ['_tight', '_medium', '_loose'],
    'spiral-arm-count': ['_1', '_2', '_3', '_4', '_5-plus', '_cant-tell'],
    'bulge-size': ['_none', '_just-noticeable', '_obvious', '_dominant'],
    'galaxy-symmetrical': ['_yes', '_no'],
    'clumps-embedded-larger-object': ['_yes', '_no']
}
# add -hubble to the end of each question
hubble_ortho_pairs = dict([(key + '-hubble', value) for key, value in hubble_pairs.items()])

# not used here, but may be helpful elsewhere
hubble_ortho_dependencies = {
    'smooth-or-featured-hubble': None,
    'how-rounded-hubble': 'smooth-or-featured-hubble_smooth',
    'clumpy-appearance-hubble': 'smooth-or-featured-hubble_features',
    'clump-count-hubble': 'clumpy-appearance-hubble_yes',
    # 'clump-configuration-hubble': ['_straight-line', '_chain', '_cluster-or-irregular', '_spiral'],
    # 'one-clump-brightest-hubble': ['_yes', '_no'],
    # 'brightest-clump-central-hubble': ['_yes', '_no'],
    # ignoring the spiral dashed line, probably rare
    'galaxy-symmetrical-hubble': 'clumpy-appearance-hubble_yes',
    'clumps-embedded-larger-object-hubble': 'clumpy-appearance-hubble_yes',
    'disk-edge-on-hubble': 'clumpy-appearance-hubble_no',
    'bulge-shape-hubble': 'disk-edge-on-hubble_yes',
    'edge-on-bulge-hubble': 'disk-edge-on-hubble_yes',
    'bar-hubble': 'disk-edge-on-hubble_no',
    'has-spiral-arms-hubble': 'disk-edge-on-hubble_no',
    'spiral-winding-hubble': 'disk-edge-on-hubble_no',
    'spiral-arm-count-hubble': 'disk-edge-on-hubble_no',
    'bulge-size-hubble': 'disk-edge-on-hubble_no'
}

hubble_ortho_questions, hubble_ortho_label_cols = label_metadata.extract_questions_and_label_cols(hubble_ortho_pairs)


if __name__ == '__main__':

    hubble_cols = [x.replace('-hubble', '') for x in hubble_ortho_label_cols]
    # print('\n'.join(cols))

    decals_cols = [x for x in label_metadata.decals_label_cols]
    # print('\n'.join(cols))

    all_cols = list(set(hubble_cols).union(decals_cols))
    all_cols.sort()
    print('\n'.join(all_cols))
