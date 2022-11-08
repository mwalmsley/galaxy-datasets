import pytest

import os

from galaxy_datasets.pytorch import datasets, galaxy_datamodule

# @pytest.fixture()
def base_root_dir():
    if os.path.isdir('/nvme1/scratch'):
        return '/nvme1/scratch/walml/repos/pytorch-galaxy-datasets/roots'
    elif os.path.isdir('/share/nas2'):
        return '/share/nas2/walml/repos/_data'
    else:
        raise FileNotFoundError

# @pytest.fixture()
def tidal_dataset():
    return datasets.TidalDataset(
        root=os.path.join(base_root_dir(), 'tidal'),
        train=True,
        download=False
    )

# @pytest.fixture()
def gz_decals_dataset():
    return datasets.GZDecalsDataset(
        root=os.path.join(base_root_dir(), 'gz_decals'),
        train=True,
        download=False
    )

def gz_desi_dataset():
    return datasets.GZDesiDataset(
        root=os.path.join(base_root_dir(), 'gz_desi'),
        train=True,
        download=False
    )

# @pytest.fixture()
def gz2_dataset():
    return datasets.GZ2Dataset(
        root=os.path.join(base_root_dir(), 'gz2'),
        download=False
    )

# @pytest.fixture()
def gz_rings_dataset():
    return datasets.GZRingsDataset(
        root=os.path.join(base_root_dir(), 'gz_rings'),
        train=True,
        download=False
    )

# @pytest.fixture()
# def legs_dataset():
#     return datasets.LegsDataset(
#         root='whatever',
#         split='train',  # slightly different API
#         download=False
#     )

# @pytest.fixture()
def gz_hubble_dataset():
    return datasets.GZHubbleDataset(
        root=os.path.join(base_root_dir(), 'gz_hubble'),
        download=False
    )


# @pytest.fixture()
def gz_candels_dataset():
    return datasets.GZCandelsDataset(
        root=os.path.join(base_root_dir(), 'gz_candels'),
        download=False
    )

# https://docs.pytest.org/en/6.2.x/fixture.html#using-marks-with-parametrized-fixtures
# pytest.param(gz_desi_dataset, marks=pytest.mark.skip)
@pytest.fixture(params=[gz2_dataset, gz_candels_dataset, gz_decals_dataset, gz_hubble_dataset, gz_rings_dataset, tidal_dataset, gz_desi_dataset])
def dataset(request):
    return request.param()


def test_dataset(dataset):

    catalog = dataset.catalog
    label_cols = dataset.label_cols

    adjusted_catalog = catalog.sample(1000)

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


# legs will fail anywhere but galahad
        


# def test_gz2_standard():

#     # gz2_dataset = 
#     catalog = gz2_dataset.catalog 

#     adjusted_catalog = catalog.sample(1000)

#     datamodule = galaxy_datamodule.GalaxyDataModule(
#         label_cols=['smooth-or-featured_smooth'],
#         catalog=adjusted_catalog
#     )

#     datamodule.prepare_data()
#     datamodule.setup()
#     for images, labels in datamodule.train_dataloader():
#         print(images.shape, labels.shape)
#         break



# def test_rings():

#     catalog = rings_dataset.catalog
#     adjusted_catalog = catalog.sample(1000)

#     datamodule = galaxy_datamodule.GalaxyDataModule(
#         label_cols=['ring_fraction'],  # counts and totals also available
#         catalog=adjusted_catalog
#     )

#     datamodule.prepare_data()
#     datamodule.setup()
#     for images, labels in datamodule.train_dataloader():
#         print(images.shape, labels.shape)
#         break
        

# def test_legs():

#     # first download is basically just a convenient way to get the images and canonical catalogs
#     # legs_dataset = 
#     legs_catalog = legs_dataset.catalog
#     adjusted_catalog = legs_catalog.sample(1000)

#     # user will probably tweak and use images/catalog directly for generic galaxy catalog datamodule
#     # (which makes its own generic datasets internally)
#     datamodule = galaxy_datamodule.GalaxyDataModule(
#         label_cols=label_metadata.decals_all_campaigns_label_cols,
#         catalog=adjusted_catalog
#     )

#     datamodule.prepare_data()
#     datamodule.setup()
#     for images, labels in datamodule.train_dataloader():
#         print(images.shape, labels.shape)
#         break
        

# def test_hubble():
#         # first download is basically just a convenient way to get the images and canonical catalogs
#     hubble_label_cols, hubble_catalog = hubble_setup(
#         root='/nvme1/scratch/walml/repos/pytorch-galaxy-datasets/roots/hubble',
#         train=True,
#         download=False
#     )
    
#     # user will probably tweak and use images/catalog directly for generic galaxy catalog datamodule
#     # (which makes its own generic datasets internally)
#     adjusted_catalog = hubble_catalog.sample(1000)
#     datamodule = galaxy_datamodule.GalaxyDataModule(
#         label_cols=hubble_label_cols,
#         catalog=adjusted_catalog
#     )

#     datamodule.prepare_data()
#     datamodule.setup()
#     for images, labels in datamodule.train_dataloader():
#         print(images.shape, labels.shape)
#         break
    

# def test_candels():
#     # first download is basically just a convenient way to get the images and canonical catalogs
#     candels_label_cols, candels_catalog = candels_setup(
#         root='/nvme1/scratch/walml/repos/pytorch-galaxy-datasets/roots/candels',
#         train=True,
#         download=False
#     )
    
#     # user will probably tweak and use images/catalog directly for generic galaxy catalog datamodule
#     # (which makes its own generic datasets internally)
#     adjusted_catalog = candels_catalog.sample(1000)
#     datamodule = galaxy_datamodule.GalaxyDataModule(
#         label_cols=candels_label_cols,
#         catalog=adjusted_catalog
#     )

#     datamodule.prepare_data()
#     datamodule.setup()
#     for images, labels in datamodule.train_dataloader():
#         print(images.shape, labels.shape)
#         break

# def test_decals():
#     # first download is basically just a convenient way to get the images and canonical catalogs

#     dr5_catalog = dr5_dataset.catalog
#     adjusted_catalog = dr5_catalog.sample(1000)

#     # user will probably tweak and use images/catalog directly for generic galaxy catalog datamodule
#     # (which makes its own generic datasets internally)
#     datamodule = galaxy_datamodule.GalaxyDataModule(
#         label_cols=label_metadata.decals_dr5_ortho_label_cols,
#         catalog=adjusted_catalog
#     )

#     datamodule.prepare_data()
#     datamodule.setup()
#     for images, labels in datamodule.train_dataloader():
#         print(images.shape, labels.shape)
#         break
