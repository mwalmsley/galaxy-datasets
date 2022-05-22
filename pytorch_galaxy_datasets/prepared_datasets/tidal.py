import os

import pandas as pd
from pytorch_galaxy_datasets import galaxy_datamodule, galaxy_dataset, download_utils

# TODO could eventually refactor this out of Zoobot as well
from zoobot.shared import label_metadata


class TidalDataset(galaxy_dataset.GalaxyDataset):
    
    def __init__(self, root, train=True, download=False, transform=None, target_transform=None, label_mode='coarse'):

        catalog, label_cols = tidal_setup(root, train, download, label_mode=label_mode)

        super().__init__(catalog, label_cols, transform, target_transform)


def tidal_setup(root, train, download, label_mode='coarse'):
    resources = [
        ('https://dl.dropboxusercontent.com/s/jq6dyc8q87h92qq/tidal_train_catalog.parquet', '39def8527823f6d4f332fbc209b15a32'),  # train catalog
        ('https://dl.dropboxusercontent.com/s/h8m7j7o3o11mt05/tidal_test_catalog.parquet', '12a8c6a23ddc8e7ad3f5bd597931bbe3'),  # test catalog
        ('https://dl.dropboxusercontent.com/s/icb6kgixe3imf4a/tidal_images.tar.gz', '1e10cd3a2a4a4c5fe39e447b2257e5f0')  # the images
    ]
    images_to_spotcheck = ['W1-2_color.jpg', 'W3-82_color.jpg', 'W3-852_color.jpg']

    downloader = download_utils.DatasetDownloader(root, resources, images_to_spotcheck)
    if download is True:
        downloader.download()

    if label_mode == 'coarse':
        label_cols = coarse_tidal_label_cols
    elif label_mode == 'finegrained':
        label_cols = finegrained_tidal_label_cols

    useful_columns = label_cols + ['filename']
    if train:
        train_catalog_loc = os.path.join(root, 'tidal_train_catalog.parquet')
        catalog = pd.read_parquet(train_catalog_loc, columns=useful_columns)
    else:
        test_catalog_loc = os.path.join(root, 'tidal_test_catalog.parquet')
        catalog = pd.read_parquet(test_catalog_loc, columns=useful_columns)

    catalog['id_str'] = catalog['filename'] # usually I have included these in the catalog itself TODO
    catalog['file_loc'] = catalog.apply(lambda x: os.path.join(root, downloader.image_dir, x['filename']), axis=1)


    return catalog, label_cols

# defined outside for use elsewhere
coarse_tidal_label_cols = ['coarse_tidal_label']
finegrained_tidal_label_cols = ['finegrained_tidal_label']

if __name__ == '__main__':

    # first download is basically just a convenient way to get the images and canonical catalogs
    tidal_catalog, label_cols = tidal_setup(
        root='/nvme1/scratch/walml/repos/pytorch-galaxy-datasets/roots/tidal',
        train=True,
        download=False,
    )
    adjusted_catalog = tidal_catalog.sample(1000)

    # user will probably tweak and use images/catalog directly for generic galaxy catalog datamodule
    # (which makes its own generic datasets internally)
    datamodule = galaxy_datamodule.GalaxyDataModule(
        label_cols=label_cols,
        catalog=adjusted_catalog
    )

    datamodule.prepare_data()
    datamodule.setup()
    for images, labels in datamodule.train_dataloader():
        print(images.shape, labels.shape)
        break
        