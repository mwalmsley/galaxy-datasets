import py
import pytest

import os

@pytest.fixture()
def example_root():
    if os.path.isdir('/share/nas2'):
        return '/share/nas2/walml/repos/_data/gz2'
    else:
        return '/nvme1/scratch/walml/repos/galaxy-datasets/roots/gz2'

@pytest.fixture()  # also a test
def get_catalog(example_root):
    from galaxy_datasets import gz2  # or gz_hubble, gz_candels, ...

    catalog, label_cols = gz2(
        root=example_root,
        train=True,
        download=True
    )
    return catalog, label_cols

def test_pytorch_dataset_custom(get_catalog):
    from galaxy_datasets.pytorch.galaxy_dataset import GalaxyDataset  # generic Dataset for galaxies

    catalog, _ = get_catalog

    dataset = GalaxyDataset(
        catalog=catalog.sample(1000),  # from gz2(...) above
        label_cols=['smooth-or-featured-gz2_smooth']
    )

def test_pytorch_dataset_canonical(example_root):

    from galaxy_datasets.pytorch import GZ2

    gz2_dataset = GZ2(
        root=example_root,
        train=True,
        download=False
    )
    image, label = gz2_dataset[0]


def test_pytorch_datamodule_custom(get_catalog):

    catalog, _ = get_catalog
    
    from galaxy_datasets.pytorch.galaxy_datamodule import GalaxyDataModule

    datamodule = GalaxyDataModule(
        label_cols=['smooth-or-featured-gz2_smooth'],
        catalog=catalog
        # optional args to specify augmentations
    )

    datamodule.prepare_data()
    datamodule.setup()
    for images, labels in datamodule.train_dataloader():
        print(images.shape, labels.shape)
        break

# currently has protobuf version error locally
@pytest.mark.skip
def test_tensorflow_dataset(get_catalog):

    import tensorflow as tf
    from galaxy_datasets.tensorflow.datasets import get_image_dataset, add_transforms_to_dataset
    from galaxy_datasets.transforms import default_transforms  # same transforms as PyTorch

    catalog, label_cols = get_catalog

    train_dataset = get_image_dataset(
        image_paths = catalog['file_loc'],
        labels=catalog[label_cols].values,
        requested_img_size=224
    )

    # specify augmentations
    transforms = default_transforms()

    # apply augmentations
    train_dataset = add_transforms_to_dataset(train_dataset, transforms)

    # batch, shuffle, prefetch for performance
    train_dataset = train_dataset.shuffle(5000).batch(64).prefetch(tf.data.experimental.AUTOTUNE)

    for images, labels in train_dataset.take(1):
        print(images.shape, labels.shape)
        break