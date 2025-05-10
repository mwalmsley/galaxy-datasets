import pytest

import os

from galaxy_datasets.pytorch import galaxy_datamodule, our_datasets


def base_root_dir():
    if os.path.isdir('/nvme1/scratch'):
        return '/nvme1/scratch/walml/repos/galaxy-datasets/roots'
    elif os.path.isdir('/share/nas2'):
        return '/share/nas2/walml/repos/_data'
    else:
        return 'roots'
    # TODO should really be relative to this file, not assuming repo root


def tidal_dataset():
    return our_datasets.Tidal(
        root=os.path.join(base_root_dir(), 'tidal'),
        train=True,
        download=False
    )


def gz_decals_dataset():
    return our_datasets.GZDecals5(
        root=os.path.join(base_root_dir(), 'gz_decals'),
        train=True,
        download=False
    )

def gz_desi_dataset():
    return our_datasets.GZDesi(
        root=os.path.join(base_root_dir(), 'gz_desi'),
        train=True,
        download=False
    )


def gz2_dataset():
    return our_datasets.GZ2(
        root=os.path.join(base_root_dir(), 'gz2'),
        download=False
    )


def gz_rings_dataset():
    return our_datasets.GZRings(
        root=os.path.join(base_root_dir(), 'gz_rings'),
        train=True,
        download=False
    )


def gz_hubble_dataset():
    return our_datasets.GZHubble(
        root=os.path.join(base_root_dir(), 'gz_hubble'),
        download=False
    )


def gz_candels_dataset():
    return our_datasets.GZCandels(
        root=os.path.join(base_root_dir(), 'gz_candels'),
        download=False
    )

def demo_rings_dataset():
    return our_datasets.DemoRings(
        root=os.path.join(base_root_dir(), 'demo_rings'),
        download=True  # tests can download, it's small
    )

# https://docs.pytest.org/en/6.2.x/fixture.html#using-marks-with-parametrized-fixtures
# pytest.param(gz_desi_dataset, marks=pytest.mark.skip)
@pytest.fixture(params=[demo_rings_dataset, gz2_dataset, gz_candels_dataset, gz_decals_dataset, gz_hubble_dataset, gz_rings_dataset, tidal_dataset, gz_desi_dataset])
# @pytest.fixture(params=[demo_rings_dataset])
def dataset(request):
    return request.param()


def test_dataset(dataset):

    catalog = dataset.catalog
    label_cols = dataset.label_cols

    adjusted_catalog = catalog.sample(256)

    # user will probably tweak and use images/catalog directly for generic galaxy catalog datamodule
    # (which makes its own generic datasets internally)
    datamodule = galaxy_datamodule.GalaxyDataModule(
        label_cols=label_cols,
        catalog=adjusted_catalog,
        batch_size=32  # need at least one batch
    )

    datamodule.prepare_data()
    datamodule.setup()

    any_batches = False
    for images, labels in datamodule.train_dataloader():
        any_batches = True
        print(images.shape, labels.shape)
        assert images.max() > (1.01 / 255.), "Image values should be in range [0, 1.], max is {}, suspected /255 twice".format(images.max())
        assert images.max() < 1.01, "Image values should be in range [0, 1. ], max is {}".format(images.max())
        assert images.min() > -0.01, "Image values should be in range [0, 1.], min is {}".format(images.min())
        break
    assert any_batches, "No batches were returned from the dataloader"