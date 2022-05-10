import pytest

import os

from pytorch_galaxy_datasets import galaxy_datamodule
from pytorch_galaxy_datasets.prepared_datasets import candels, dr5, gz2, legs, rings, tidal


def test_tidal_datamodule():

    # first download is basically just a convenient way to get the images and canonical catalogs
    label_cols, catalog = tidal.tidal_setup(
        root='/nvme1/scratch/walml/repos/pytorch-galaxy-datasets/roots/tidal_root',
        train=True,
        download=False
    )
    adjusted_catalog = catalog.sample(1000)
    datamodule = galaxy_datamodule.GalaxyDataModule(
        label_cols=label_cols,  # counts and totals also available
        catalog=adjusted_catalog
    )

    datamodule.prepare_data()
    datamodule.setup()
    for images, labels in datamodule.train_dataloader():
        print(images.shape, labels.shape)
        break
        


def test_rings_datamodule():

    # first download is basically just a convenient way to get the images and canonical catalogs
    label_cols, catalog = rings.rings_setup(
        root='/nvme1/scratch/walml/repos/pytorch-galaxy-datasets/roots/rings_root',
        train=True,
        download=False
    )
    adjusted_catalog = catalog.sample(1000)
    datamodule = galaxy_datamodule.GalaxyDataModule(
        label_cols=label_cols,  # counts and totals also available
        catalog=adjusted_catalog
    )

    datamodule.prepare_data()
    datamodule.setup()
    for images, labels in datamodule.train_dataloader():
        print(images.shape, labels.shape)
        break
        

def test_candels_datamodule():

    # first download is basically just a convenient way to get the images and canonical catalogs
    label_cols, catalog = candels.candels_setup(
        root='/nvme1/scratch/walml/repos/pytorch-galaxy-datasets/roots/candels_root',
        train=True,
        download=False
    )
    adjusted_catalog = catalog.sample(1000)
    datamodule = galaxy_datamodule.GalaxyDataModule(
        label_cols=label_cols,  # counts and totals also available
        catalog=adjusted_catalog
    )

    datamodule.prepare_data()
    datamodule.setup()
    for images, labels in datamodule.train_dataloader():
        print(images.shape, labels.shape)
        break
        

def test_dr5_datamodule():

    # first download is basically just a convenient way to get the images and canonical catalogs
    label_cols, catalog = dr5.decals_dr5_setup(
        root='/nvme1/scratch/walml/repos/pytorch-galaxy-datasets/roots/dr5_root',
        train=True,
        download=False
    )
    adjusted_catalog = catalog.sample(1000)
    datamodule = galaxy_datamodule.GalaxyDataModule(
        label_cols=label_cols,  # counts and totals also available
        catalog=adjusted_catalog
    )

    datamodule.prepare_data()
    datamodule.setup()
    for images, labels in datamodule.train_dataloader():
        print(images.shape, labels.shape)
        break


@pytest.mark.skipif(not os.path.isdir('/share/nas2/walml'), reason="Data only exists on Galahad")
def test_legs_datamodule():

    # first download is basically just a convenient way to get the images and canonical catalogs
    label_cols, catalog = legs.legs_setup(
        # root='/nvme1/scratch/walml/repos/pytorch-galaxy-datasets/roots/legs_root',
        train=True,
        download=False
    )
    adjusted_catalog = catalog.sample(1000)
    datamodule = galaxy_datamodule.GalaxyDataModule(
        label_cols=label_cols,  # counts and totals also available
        catalog=adjusted_catalog
    )

    datamodule.prepare_data()
    datamodule.setup()
    for images, labels in datamodule.train_dataloader():
        print(images.shape, labels.shape)
        break
