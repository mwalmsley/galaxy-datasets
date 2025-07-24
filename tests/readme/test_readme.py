import py
import pytest

import os

@pytest.fixture()
def base_root_dir():
    if os.path.isdir('/nvme1/scratch'):
        return '/nvme1/scratch/walml/repos/galaxy-datasets/roots'
    elif os.path.isdir('/share/nas2'):
        return '/share/nas2/walml/repos/_data'
    else:
        return 'roots'
    # TODO should really be relative to this file, not assuming repo root


@pytest.fixture()  # also a test
def get_catalog(base_root_dir):
    from galaxy_datasets import gz2  # or gz_hubble, gz_candels, ...

    catalog, label_cols = gz2(
        root=base_root_dir,
        train=True,
        download=True
    )
    return catalog, label_cols

def test_pytorch_dataset_custom(get_catalog):
    from galaxy_datasets.pytorch.galaxy_dataset import CatalogDataset  # generic Dataset for galaxies

    catalog, _ = get_catalog

    dataset = CatalogDataset(
        catalog=catalog.sample(1000),  # from gz2(...) above
        label_cols=['smooth-or-featured-gz2_smooth']
    )

def test_pytorch_dataset_canonical(base_root_dir):

    from galaxy_datasets.pytorch import GZ2

    gz2_dataset = GZ2(
        root=base_root_dir,
        train=True,
        download=False
    )
    batch = gz2_dataset[0]
    image = batch['image']
    label = batch['smooth-or-featured-gz2_smooth']


def test_pytorch_datamodule_custom(get_catalog):

    catalog, _ = get_catalog
    
    from galaxy_datasets.pytorch.galaxy_datamodule import CatalogDataModule
    from galaxy_datasets.transforms import get_galaxy_transform, default_view_config

    datamodule = CatalogDataModule(
        label_cols=['smooth-or-featured-gz2_smooth'],
        catalog=catalog,
        train_transform=get_galaxy_transform(default_view_config()),
        test_transform=get_galaxy_transform(default_view_config())
    )

    datamodule.prepare_data()
    datamodule.setup()
    for batch in datamodule.train_dataloader():
        images = batch['image']
        labels = batch['smooth-or-featured-gz2_smooth']
        print(images.shape, labels.shape)
        break
