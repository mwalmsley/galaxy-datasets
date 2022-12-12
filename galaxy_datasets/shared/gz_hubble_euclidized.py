import os
import logging

import pandas as pd

# this one has label_metadata contained in this repo, not zoobot. Others will follow, perhaps.
from zoobot.shared import label_metadata

from galaxy_datasets.shared import download_utils


def gz_hubble_euclidized(root, train, download):
    logging.info('Setting up gz_hubble_euclidized dataset')
    resources = [
        ('https://drive.google.com/file/d/1Vm__dpPbyojLFi58RTTZoyzmFSWv9Aup/view?usp=share_link', None),  # train catalog
        ('https://drive.google.com/file/d/1Vm__dpPbyojLFi58RTTZoyzmFSWv9Aup/view?usp=share_link',None),  # test catalog
        ('https://drive.google.com/file/d/1-6mwYz0Aa3Demd594dWr_1Gi3YTCkoIa/view?usp=share_link', None)  # the images 'b6fed2463bb2d17ddb8302f6b060534a'
    ]
    images_to_spotcheck = ['20163083.jpg']

    downloader = download_utils.DatasetDownloader(root, resources, images_to_spotcheck)
    if download is True:
        downloader.download()

    # label_cols = [question + answer for question, answers in hubble_ortho_pairs.items() for answer in answers]  # defined below, globally in this script (for imports elsewhere)
    label_cols = hubble_ortho_label_cols  # see below

    useful_columns = label_cols + ['filename']
    if train:
        train_catalog_loc = os.path.join(root, 'catalog_hubble_bright.csv')
        catalog = pd.read_parquet(train_catalog_loc, columns=useful_columns)
    else:
        test_catalog_loc = os.path.join(root,  'catalog_hubble_bright.csv')
        catalog = pd.read_parquet(test_catalog_loc, columns=useful_columns)

    catalog['file_loc'] = catalog.apply(lambda x: os.path.join(downloader.image_dir, x['filename']), axis=1)  # no subfolder

    logging.info('gz_euclid dataset ready')
    return catalog, label_cols

# TODO may change features to featured-or-disk
hubble_pairs = {
    'smooth-or-features': ['_smooth','_features-or-disk','_star-or-artifact'],
    'disk-edge-on': ['_yes','_no'],
    'bar': ['_bar','_no-bar'],
    'spiral': ['_spiral','_no-spiral'],
    'bulge-prominence': ['_no-bulge','_just-noticable','_obvious','_dominant'],
    'odd': ['_yes','_no'],
    'rounded': ['_completely-round','_in-between','_cigar-shaped'],
    'odd-feature': ['_ring','_lens-or-arc','_disturbed','_irregular','_other','_merger','_dust-lane'],
    'bulge-shape': ['_rounded','_boxy','_no-bulge'],
    'arms-winding': ['_tight','_medium','_loose'],
    'arms-number': ['_1','_2','_3','_4','_more-than-4','_cant_tell'],
    'clumpy': ['_yes','_no'],
    'bright-clump': ['_yes','_no'],
    'bright-clump-central': ['_yes','_no'],
    'clumps-arrangement': ['_line','_chain','_cluster','_spiral'],
    'clumps-count': ['_1','_2','_3','_4','_more-than-4','_cant-tell'],
    'clumps-symmetrical': ['_yes','_no'],
    'clumps-embedded': ['_yes','_no'],
}
# add -hubble to the end of each question
hubble_ortho_pairs = dict([(key + '-hubble', value) for key, value in hubble_pairs.items()])

# not used here, but may be helpful elsewhere
hubble_ortho_dependencies = {
    'smooth-or-features': None,
    'disk-edge-on': 'clumpy_no',
    'bar': 'disk-edge-on_no',
    'spiral': 'disk-edge-on_no',
    'bulge-prominence': 'disk-edge-on_no',
    'odd': None,
    'rounded': 'smooth-or-features_smooth',
    'odd-feature': 'odd_yes',
    'bulge-shape': 'disk-edge-on_yes',
    'arms-winding': 'spiral_spiral',
    'arms-number': 'spiral_spiral',
    'clumpy': 'smooth-or-features_features-or-disk',
    'bright-clump': 'clumpy_yes', #TODO needs to be checked
    'bright-clump-central': 'bright-clump_yes', 
    'clumps-arrangement': 'clumpy_yes', #TODO needs to be checked
    'clumps-count': 'clumpy_yes',
    'clumps-symmetrical': 'clumpy_yes',
    'clumps-embedded': 'clumpy_yes',
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
