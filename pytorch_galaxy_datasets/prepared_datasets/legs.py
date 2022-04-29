
import pandas as pd
from pytorch_galaxy_datasets import galaxy_datamodule, galaxy_dataset

# TODO could eventually refactor this out of Zoobot as well
from zoobot.shared import label_metadata


class LegsDataset(galaxy_dataset.GalaxyDataset):
    
    def __init__(self, data_dir=None, train=True, download=False, transform=None, target_transform=None):

        label_cols, catalog = self.legs_setup(data_dir, train, download)

        # paths are not adjusted as cannot be downloaded
        # catalog = _temp_adjust_catalog_paths(catalog)
        # catalog = adjust_catalog_dtypes(catalog, label_cols)

        super().__init__(catalog, label_cols, transform, target_transform)


def legs_setup(data_dir, train, download):
    if data_dir is not None:
        'Legacy Survey cannot be downloaded - ignoring data_dir {}'.format(data_dir)

    if download is True:
        # downloader.download()
        raise NotImplementedError('Legacy Survey cannot be downloaded')

    label_cols = label_metadata.decals_all_campaigns_ortho_pairs

    usecols = label_cols + ['file_loc']
    if train:
        train_catalog_loc = '/share/nas2/walml/repos/gz-decals-classifiers/data/decals/presplit_catalogs/legs_all_campaigns_ortho_train_catalog.parquet'
        catalog = pd.read_parquet(train_catalog_loc, usecols=usecols)
    else:
        test_catalog_loc = '/share/nas2/walml/repos/gz-decals-classifiers/data/decals/presplit_catalogs/legs_all_campaigns_ortho_test_catalog.parquet'
        catalog = pd.read_csv(test_catalog_loc, usecols=usecols)
    return label_cols,catalog




if __name__ == '__main__':


    # first download is basically just a convenient way to get the images and canonical catalogs
    legs_dataset = LegsDataset(
        data_dir='whatever',
        train=True,
        download=False  # will fail except on galahad
    )
    legs_catalog = legs_dataset.catalog
    adjusted_catalog = legs_catalog.sample(1000)

    # user will probably tweak and use images/catalog directly for generic galaxy catalog datamodule
    # (which makes its own generic datasets internally)
    datamodule = galaxy_datamodule.GalaxyDataModule(
        label_cols=label_metadata.decals_all_campaigns_label_cols,
        catalog=adjusted_catalog
    )

    datamodule.prepare_data()
    datamodule.setup()
    for images, labels in datamodule.train_dataloader():
        print(images.shape, labels.shape)
        break
        
