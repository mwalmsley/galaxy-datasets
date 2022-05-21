
import os

import pandas as pd
from pytorch_galaxy_datasets import galaxy_dataset, galaxy_datamodule, download_utils

# this one has label_metadata contained in this repo, not zoobot. Others will follow, perhaps.
from zoobot.shared import label_metadata



class CandelsDataset(galaxy_dataset.GalaxyDataset):
    
    def __init__(self, root, train=True, download=False, transform=None, target_transform=None):

        catalog, label_cols = candels_setup(root, train, download)

        super().__init__(catalog, label_cols, transform, target_transform)


def candels_setup(root, train, download):
    resources = [
        ('https://dl.dropboxusercontent.com/s/cnjvdinnhh1r1md/candels_ortho_train_catalog.parquet', '90593d1bab79a608cf0e645d6fd8e741'),  # train catalog
        ('https://dl.dropboxusercontent.com/s/y83v1gktw72hs0f/candels_ortho_test_catalog.parquet', '1062993dd8df09684b335678ab3fa8e3'),  # test catalog
        ('https://dl.dropboxusercontent.com/s/d67we9xsn8vyr5k/candels_images.tar.gz', 'b621ee4e650cf084a1a0c1fe5c9d0a21')  # the images
    ]
    images_to_spotcheck = ['COS_9933.jpg', 'UDS_21986.jpg', 'GDS_9405.jpg']

    downloader = download_utils.DatasetDownloader(root, resources, images_to_spotcheck)
    if download is True:
        downloader.download()

    # label_cols = [question + answer for question, answers in candels_ortho_pairs.items() for answer in answers]  # defined below, globally in this script (for imports elsewhere)
    label_cols = candels_ortho_label_cols  # see below

    useful_columns = label_cols + ['filename']
    if train:
        train_catalog_loc = os.path.join(root, 'candels_ortho_train_catalog.parquet')
        catalog = pd.read_parquet(train_catalog_loc, columns=useful_columns)
    else:
        test_catalog_loc = os.path.join(root, 'candels_ortho_test_catalog.parquet')
        catalog = pd.read_parquet(test_catalog_loc, columns=useful_columns)

    catalog['file_loc'] = catalog.apply(lambda x: os.path.join(root, downloader.image_dir, x['filename']), axis=1)  # no subfolder

    return catalog, label_cols

# TODO may change features to featured-or-disk
candels_pairs = {
    'smooth-or-featured': ['_smooth', '_features', '_artifact'],
    'how-rounded': ['_completely', '_in-between', '_cigar-shaped'],
    'clumpy-appearance': ['_yes', '_no'],
    'clump-count': ['_1', '_2', '_3', '_4', '_5-plus', '_cant-tell'],
    # disable these for now as I don't support having several but not all answers leading to the same next question
    # 'clump-configuration': ['_straight-line', '_chain', '_cluster-or-irregular', '_spiral'],
    # 'one-clump-brightest': ['_yes', '_no'],
    # 'brightest-clump-central': ['_yes', '_no'],
    # 'galaxy-symmetrical': ['_yes', '_no'],
    # 'clumps-embedded-larger-object': ['_yes', '_no'],
    'disk-edge-on': ['_yes', '_no'],
    'edge-on-bulge': ['_yes', '_no'],
    'bar': ['_yes', '_no'],
    'has-spiral-arms': ['_yes', '_no'],
    'spiral-winding': ['_tight', '_medium', '_loose'],
    'spiral-arm-count': ['_1', '_2', '_3', '_4', '_5-plus', '_cant-tell'],
    'bulge-size': ['_none', '_obvious', '_dominant'],
    'merging': ['_merger', '_tidal-debris', '_both', '_neither']
}
# add -candels to the end of each question
candels_ortho_pairs = dict([(key + '-candels', value) for key, value in candels_pairs.items()])

# not used here, but may be helpful elsewhere
candels_ortho_dependencies = {
    'smooth-or-featured-candels': None,
    'how-rounded-candels': 'smooth-or-featured-candels_smooth',
    'clumpy-appearance-candels': 'smooth-or-featured-candels_features',
    'clump-count-candels': 'clumpy-appearance-candels_yes',
    # 'clump-configuration-candels': ['_straight-line', '_chain', '_cluster-or-irregular', '_spiral'],
    # 'one-clump-brightest-candels': ['_yes', '_no'],
    # 'brightest-clump-central-candels': ['_yes', '_no'],
    # 'galaxy-symmetrical-candels': ['_yes', '_no'],
    # 'clumps-embedded-larger-object-candels': ['_yes', '_no'],
    'disk-edge-on-candels': 'clumpy-appearance-candels_no',
    'edge-on-bulge-candels': 'disk-edge-on-candels_yes',
    'bar-candels': 'disk-edge-on-candels_no',
    'has-spiral-arms-candels': 'disk-edge-on-candels_no',
    'spiral-winding-candels': 'disk-edge-on-candels_no',
    'spiral-arm-count-candels': 'disk-edge-on-candels_no',
    'bulge-size-candels': 'disk-edge-on-candels_no',
    'merging-candels': None
}

candels_ortho_questions, candels_ortho_label_cols = label_metadata.extract_questions_and_label_cols(candels_ortho_pairs)


if __name__ == '__main__':

    # first download is basically just a convenient way to get the images and canonical catalogs
    candels_label_cols, candels_catalog = candels_setup(
        root='/nvme1/scratch/walml/repos/pytorch-galaxy-datasets/roots/candels',
        train=True,
        download=False
    )
    
    # user will probably tweak and use images/catalog directly for generic galaxy catalog datamodule
    # (which makes its own generic datasets internally)
    adjusted_catalog = candels_catalog.sample(1000)
    datamodule = galaxy_datamodule.GalaxyDataModule(
        label_cols=candels_label_cols,
        catalog=adjusted_catalog
    )

    datamodule.prepare_data()
    datamodule.setup()
    for images, labels in datamodule.train_dataloader():
        print(images.shape, labels.shape)
        break
    