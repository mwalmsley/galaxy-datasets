import pytest

import os

from pytorch_galaxy_datasets import galaxy_datamodule
from pytorch_galaxy_datasets.prepared_datasets import candels, dr5, gz2, legs, rings, tidal

@pytest.fixture
def base_root_dir():
    if os.path.isdir('/nvme1/scratch'):
        return '/nvme1/scratch/walml/repos/pytorch-galaxy-datasets/roots'
    elif os.path.isdir('/share/nas2'):
        return '/share/nas2/walml/repos/_data'
    else:
        raise FileNotFoundError


def test_tidal_datamodule(base_root_dir):

    # first download is basically just a convenient way to get the images and canonical catalogs
    catalog, label_cols = tidal.tidal_setup(
        root=os.path.join(base_root_dir, 'tidal'),
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
        


def test_rings_datamodule(base_root_dir):

    # first download is basically just a convenient way to get the images and canonical catalogs
    catalog, label_cols = rings.rings_setup(
        root=os.path.join(base_root_dir, 'rings'),
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
        

def test_candels_datamodule(base_root_dir):

    # first download is basically just a convenient way to get the images and canonical catalogs
    catalog, label_cols = candels.candels_setup(
        root=os.path.join(base_root_dir, 'candels'),
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
        

def test_dr5_datamodule(base_root_dir):

    # first download is basically just a convenient way to get the images and canonical catalogs
    catalog, label_cols = dr5.decals_dr5_setup(
        root=os.path.join(base_root_dir, 'decals_dr5'),
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
    catalog, label_cols = legs.legs_setup(
        # root=os.path.join(base_root_dir, '/nvme1/scratch/walml/repos/pytorch-galaxy-datasets/roots/legs',
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
