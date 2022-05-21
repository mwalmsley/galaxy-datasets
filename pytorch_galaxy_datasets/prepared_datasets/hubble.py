
import os

import pandas as pd
from pytorch_galaxy_datasets import galaxy_dataset, galaxy_datamodule, download_utils

# this one has label_metadata contained in this repo, not zoobot. Others will follow, perhaps.
from zoobot.shared import label_metadata



class HubbleDataset(galaxy_dataset.GalaxyDataset):
    
    def __init__(self, root, train=True, download=False, transform=None, target_transform=None):

        catalog, label_cols = hubble_setup(root, train, download)

        super().__init__(catalog, label_cols, transform, target_transform)


def hubble_setup(root, train, download):
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
    label_cols = hubble_ortho_label_cols  # see below

    useful_columns = label_cols + ['filename']
    if train:
        train_catalog_loc = os.path.join(root, 'hubble_ortho_train_catalog.parquet')
        catalog = pd.read_parquet(train_catalog_loc, columns=useful_columns)
    else:
        test_catalog_loc = os.path.join(root, 'hubble_ortho_test_catalog.parquet')
        catalog = pd.read_parquet(test_catalog_loc, columns=useful_columns)

    catalog['file_loc'] = catalog.apply(lambda x: os.path.join(root, downloader.image_dir, x['filename']), axis=1)  # no subfolder

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
    # 'galaxy-symmetrical': ['_yes', '_no'],
    # 'clumps-embedded-larger-object': ['_yes', '_no'],
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

    # first download is basically just a convenient way to get the images and canonical catalogs
    hubble_label_cols, hubble_catalog = hubble_setup(
        root='/nvme1/scratch/walml/repos/pytorch-galaxy-datasets/roots/hubble',
        train=True,
        download=False
    )
    
    # user will probably tweak and use images/catalog directly for generic galaxy catalog datamodule
    # (which makes its own generic datasets internally)
    adjusted_catalog = hubble_catalog.sample(1000)
    datamodule = galaxy_datamodule.GalaxyDataModule(
        label_cols=hubble_label_cols,
        catalog=adjusted_catalog
    )

    datamodule.prepare_data()
    datamodule.setup()
    for images, labels in datamodule.train_dataloader():
        print(images.shape, labels.shape)
        break
    