import os

import pandas as pd
from pytorch_galaxy_datasets import galaxy_datamodule, galaxy_dataset, download_utils

# TODO could eventually refactor this out of Zoobot as well
from zoobot.shared import label_metadata

# DR8 will be basically the same


class DecalsDR5Dataset(galaxy_dataset.GalaxyDataset):
    
    def __init__(self, root, train=True, download=False, transform=None, target_transform=None):

        catalog, label_cols = decals_dr5_setup(root, train, download)

        super().__init__(catalog, label_cols, transform, target_transform)


def decals_dr5_setup(root, train, download):
    resources = [
        ('https://dl.dropboxusercontent.com/s/1tuehajonhgv8a2/decals_dr5_ortho_train_catalog.parquet', 'a0cd74edc073fdff068370f6eefeb802'),  # train catalog
        ('https://dl.dropboxusercontent.com/s/3vo6hjlnbqgzuxz/decals_dr5_ortho_test_catalog.parquet', '55820e3712b22e587f6971e4b6c73dfe'),  # test catalog
        ('https://dl.dropboxusercontent.com/s/bs6jp0mkgiekhww/decals_dr5_images.tar.gz', '1347de4c8df4ec579d5a58241c1f280b')  # the images
    ]
    images_to_spotcheck = ['J073/J073013.60+242930.0.jpg']

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

    catalog['file_loc'] = catalog.apply(lambda x: os.path.join(root, downloader.image_dir, x['subfolder'], x['filename']), axis=1)

    # default, but not actually used here
    label_cols = label_metadata.decals_dr5_ortho_label_cols
    return catalog, label_cols



# def _temp_adjust_catalog_dtypes(catalog, label_cols):
#     # enforce datatypes
#     for answer_col in label_cols:
#         catalog[answer_col] = catalog[answer_col].astype(int)
#     return catalog


if __name__ == '__main__':

    # first download is basically just a convenient way to get the images and canonical catalogs
    dr5_dataset = DecalsDR5Dataset(
        root='/nvme1/scratch/walml/repos/pytorch-galaxy-datasets/roots/decals_dr5',
        train=True,
        download=False
    )
    dr5_catalog = dr5_dataset.catalog
    adjusted_catalog = dr5_catalog.sample(1000)

    # user will probably tweak and use images/catalog directly for generic galaxy catalog datamodule
    # (which makes its own generic datasets internally)
    datamodule = galaxy_datamodule.GalaxyDataModule(
        label_cols=label_metadata.decals_dr5_ortho_label_cols,
        catalog=adjusted_catalog
    )

    datamodule.prepare_data()
    datamodule.setup()
    for images, labels in datamodule.train_dataloader():
        print(images.shape, labels.shape)
        break
        

# class DECaLSDR5DataModule(galaxy_datamodule.GalaxyDataModule):

#     def __init__(self, *args, **kwargs) -> None:
#         """
#         Currently identical to GalaxyDataModule - see that description
#         """
#         super().__init__(*args, dataset_class=DECaLSDR5Dataset, **kwargs)

#     def prepare_data(self):
#         DECaLSDR5Dataset(self.root, download=True)
