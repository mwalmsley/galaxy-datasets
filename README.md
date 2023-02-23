# galaxy-datasets

ML-friendly datasets for major Galaxy Zoo citizen science campaigns.

- PyTorch Datasets and PyTorch Lightning DataModules
- TensorFlow tf.data.Dataset's
- Framework-independent (i.e. TensorFlow-friendly) download and augmentation code

You may also be interested in [Galaxy MNIST](https://github.com/mwalmsley/galaxy_mnist) as a simple dataset for teaching/debugging.
 

| Name      | Method | PyTorch Dataset | Published | Downloadable | Galaxies
| ----------- | ----- | ----------- | --- | ---- | ---- |
| Galaxy Zoo 2 | gz2 | GZ2 | &#x2611; | &#x2611; | ~210k (main sample) |
| GZ Hubble*   | gz_hubble | GZHubble | &#x2611; | &#x2611; | ~106k (main sample) |
| GZ CANDELS   | gz_candels | GZCandels | &#x2611; | &#x2611; | ~50k |
| GZ DECaLS GZD-5 | gz_decals_5 | GZDecals5 | &#x2611; | &#x2611; | ~230k (GZD-5 only)|
| GZ Rings | gz_rings | GZRings | &#x2612; | &#x2611; | ~93k |
| GZ DESI  | gz_desi | GZDesi | &#x2612; | WIP | WIP |
| CFHT Tidal* | tidal | Tidal | &#x2611; | &#x2611; | 1760 (expert) |

Any datasets marked as downloadable but not marked as published are only downloadable internally (for development purposes).

For each dataset, you must cite/acknowledge the GZ data release paper and the original telescope survey from which the images were derived. See [data.galaxyzoo.org](data.galaxyzoo.org) for the data release paper citations to use.

*GZ Hubble is also available in "euclidised" form (i.e. with the Euclid PSF applied) to Euclid collaboration members. The method is `gz_hubble_euclidised`. Courtesy of Ben Aussel.

**CFHT Tidal is not a Galaxy Zoo dataset, but rather a small expert-labelled dataset of tidal features from [Atkinson 2013](https://doi.org/10.1088/0004-637X/765/1/28).
MW reproduced and modified the images in [Walmsley 2019](https://doi.org/10.1093/mnras/sty3232). We include it here as a challenging fine-grained morphology classification task with little labelled data.

## Installation

Installing [zoobot](www.github/mwalmsley/zoobot) will automatically install this package as a dependency.

To install directly:

- `pip install galaxy-datasets[pytorch]` for PyTorch dependencies
- `pip install galaxy-datasets[tensorflow]` for TensorFlow dependencies
- `pip install galaxy-datasets[pytorch,tensorflow]` for both

For local development (e.g. adding a new dataset), you can install this by cloning from github, then running `pip install -e .` in the cloned repo root. This makes changing the code easier than if you don't use the -e, in which case the package is installed under sitepackages.

I suggest either:

- For basic use without changes, installing `zoobot` via pip and allowing pip to manage this dependency
- For development, installing both `zoobot` and `galaxy-datasets` via git

## Usage

Check out the PyTorch quickstart Colab [here](https://colab.research.google.com/drive/1mLXz0tUWO_kDrfWTlxB7JT2AnPPWQODg?usp=sharing), or keep reading for more explanation.

### Framework-Independent

To download a dataset:

    from galaxy_datasets import gz2  # or gz_hubble, gz_candels, ...

    catalog, label_cols = gz2(
        root='your_data_folder/gz2',
        train=True,
        download=True
    )

This will download the images and train/test catalogs to `root`. Each `catalog` is a pandas DataFrame with the column `file_loc` giving absolute image paths and additional columns `label_cols = ['col_a', 'col_b', ...]` giving the labels (usually, the number of volunteers who gave each answer for each galaxy). If `train=True`, the method returns the train catalog, otherwise, the test catalog.

If training Zoobot from scratch, this is all you need. For example, in PyTorch:

    from zoobot.pytorch.training import train_with_pytorch_lightning

    train_with_pytorch_lightning.train_default_zoobot_from_scratch(
        catalog=catalog,
        save_dir=save_dir,
        schema=gz2_schema, # see zoobot/pytorch/examples/minimal_example.py
        ...
    )

Otherwise, you might like to use the classes in this package to load these catalogs into ML-friendly inputs.

### PyTorch

Create a PyTorch Dataset from a catalog like so:

    from galaxy_datasets.pytorch.galaxy_dataset import GalaxyDataset  # generic Dataset for galaxies

    dataset = GalaxyDataset(
        catalog=catalog.sample(1000),  # from gz2(...) above
        label_cols=['smooth-or-featured-gz2_smooth'],
        transform=optional_transforms_if_you_like
    )

Notice how you can adjust the catalog before creating the Dataset. This gives flexibility to try training on e.g. different catalog subsets.

If you don't want to change anything about the catalog, you can skip the framework-independent download and use a named class from `galaxy_datasets.pytorch`, which takes the same arguments and directly gives a Dataset:

    from galaxy_datasets.pytorch import GZ2

    gz2_dataset = GZ2(
        root='your_data_folder/gz2',
        train=True,
        download=False
    )
    image, label = gz2_dataset[0]
    plt.imshow(image)

You might also find the PyTorch Lightning DataModule under `galaxy_datasets/pytorch/galaxy_datamodule` useful. Zoobot uses this for training and finetuning.

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

### TensorFlow

To create a tf.data.Dataset from a catalog:

    import tensorflow as tf
    from galaxy_datasets.tensorflow.datasets import get_image_dataset, add_transforms_to_dataset
    from galaxy_datasets.transforms import default_transforms  # same transforms as PyTorch

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

## Download Notes

Datasets are downloaded like:

- {root}
    - images
        - subfolder (except GZ2)
            - image.jpg
    - {catalog_name(s)}.parquet

The whole dataset is downloaded regardless of whether `train=True` or `train=False`.
