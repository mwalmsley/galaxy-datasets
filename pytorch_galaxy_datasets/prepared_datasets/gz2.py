import os

import pandas as pd
from pytorch_galaxy_datasets import galaxy_datamodule, galaxy_dataset, download_utils

# TODO could eventually refactor this out of Zoobot as well
from zoobot.shared import label_metadata


# TODO add train flag? currently full dataset
class GZ2Dataset(galaxy_dataset.GalaxyDataset):
    
    def __init__(self, data_dir, download=False, transform=None, target_transform=None):

        catalog, label_cols = gz2_setup(data_dir, download)  # no train arg

        super().__init__(catalog, label_cols, transform, target_transform)


def gz2_setup(data_dir, download):
    resources = [
        ('https://dl.dropboxusercontent.com/s/fhp3o4jsdvx8r7y/gz2_downloadable_catalog.parquet.gz', 'e0d74efc0a8a2f99c789817015f8e688'),  # the catalog
        ('https://zenodo.org/record/3565489/files/images_gz2.zip', 'bc647032d31e50c798770cf4430525c7')  # the images
    ]
    images_to_spotcheck = ['100097.jpg']  # TODO could be programatic of course

    downloader = download_utils.DatasetDownloader(data_dir, resources, images_to_spotcheck)
    if download is True:
        downloader.download()
    
    catalog = pd.read_parquet(os.path.join(data_dir, 'gz2_downloadable_catalog.parquet'))
    catalog['file_loc'] = catalog['filename'].apply(
        lambda x: os.path.join(downloader.image_dir, x))

    label_cols = label_metadata.gz2_label_cols
    return catalog, label_cols


if __name__ == '__main__':

    # can use this directly, e.g. to visualise the dataset
    gz2_dataset = GZ2Dataset(
        data_dir='/nvme1/scratch/walml/repos/pytorch-galaxy-datasets/tests/gz2_root',
        download=False
    )
    gz2_catalog = gz2_dataset.catalog  # and can get the catalog/cols for tweaking
    gz2_label_cols = gz2_dataset.label_cols

    # or, can use the setup method directly
    catalog, label_cols = gz2_setup(
        data_dir='/nvme1/scratch/walml/repos/pytorch-galaxy-datasets/tests/gz2_root',
        download=False
    )
    adjusted_catalog = gz2_catalog.sample(1000)

    datamodule = galaxy_datamodule.GalaxyDataModule(
        label_cols=['smooth-or-featured_smooth'],
        catalog=adjusted_catalog
    )

    datamodule.prepare_data()
    datamodule.setup()
    for images, labels in datamodule.train_dataloader():
        print(images.shape, labels.shape)
        break