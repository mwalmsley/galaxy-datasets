# pytorch-galaxy-datasets
PyTorch Datasets and PyTorch Lightning Datamodules for loading images and labels from Galaxy Zoo citizen science campaigns.

| Name      | Class | Published | Downloadable | Galaxies
| ----------- | ----------- | --- | ---- | ---- |
| Galaxy Zoo 2 | GZ2 | &#x2611; | &#x2611; | ~210k (main sample) |
| GZ Hubble   | Hubble | &#x2611; | &#x2611; | ~106k (main sample) |
| GZ CANDELS   | Candels | &#x2611; | &#x2611; | ~50k |
| GZ DECaLS GZD-5   | DecalsDR5 | &#x2611; | &#x2611; | ~230k |
| Galaxy Zoo Rings | Rings | &#x2612; | &#x2611; | ~93k |
| GZ Legacy Survey  | Legs | &#x2612; | z < 0.1 only | ~375k + 8.3m unlabelled |
| CFHT Tidal* | Tidal | &#x2611; | &#x2611; | 1760 (expert) |

Any datasets marked as downloadable but not marked as published are only downloadable internally (for development purposes).

If a dataset is published but not marked as downloadable (none currently), it means I haven't yet got around to making the download automatic. You can still download it via the paper instructions.

You may also be interested in [Galaxy MNIST](https://github.com/mwalmsley/galaxy_mnist) as a simple dataset for teaching/debugging.

For each dataset, you must cite/acknowledge the GZ data release paper and the original telescope survey from which the images were derived. See [data.galaxyzoo.org](data.galaxyzoo.org) for the data release paper citations to use.

*CFHT Tidal is not a Galaxy Zoo dataset, but rather a small expert-labelled dataset of tidal features from [Atkinson 2013](https://doi.org/10.1088/0004-637X/765/1/28).
MW reproduced and modified the images in [Walmsley 2019](https://doi.org/10.1093/mnras/sty3232).
We include it here as a challenging fine-grained morphology classification task with little labelled data.

### Installation

For local development (e.g. adding a new dataset), you can install this by cloning from github, then running `pip install -e .` in the cloned repo root. 

It will also be installed by default as a dependency of `zoobot` if you specify the pytorch version of `zoobot` - but this is slightly trickier if you'd like to make changes as it'll be installed under your `sitepackages`.

### Usage

You can load each prepared dataset as a pytorch Dataset like so:

    from pytorch_galaxy_datasets.prepared_datasets import GZ2Dataset

    gz2_dataset = GZ2Dataset(
        root='/nvme1/scratch/walml/repos/pytorch-galaxy-datasets/roots/gz2',
        train=True,
        download=False
    )
    image, label = gz2_dataset[0]
    plt.imshow(image)

You will probably want to customise the dataset, selecting a subset of galaxies or labels. Do this with the `{dataset}_setup()` methods.

    from pytorch_galaxy_datasets.prepared_datasets import gz2_setup

    catalog, label_cols = gz2_setup(
        root='/nvme1/scratch/walml/repos/pytorch-galaxy-datasets/roots/gz2',
        train=True,
        download=False
    )
    adjusted_catalog = gz2_catalog.sample(1000)

You can then customise the catalog and labels before creating a generic GalaxyDataset, which can be used with your own transforms etc. like any other pytorch dataset

    from pytorch_galaxy_datasets.galaxy_dataset import GalaxyDataset

    dataset = GalaxyDataset(
        label_cols=['smooth-or-featured_smooth'],
        catalog=adjusted_catalog,
        transforms=some_torchvision_transforms_if_you_like
    )

For training models, I recommend using Pytorch Lightning and GalaxyDataModule, which has default transforms for supervised learning.

    from pytorch_galaxy_datasets.galaxy_datamodule import GalaxyDataModule

    datamodule = GalaxyDataModule(
        label_cols=['smooth-or-featured_smooth'],
        catalog=adjusted_catalog
    )

    datamodule.prepare_data()
    datamodule.setup()
    for images, labels in datamodule.train_dataloader():
        print(images.shape, labels.shape)
        break

You can also get the canonical catalog and label_cols from the Dataset, if you prefer.

    gz2_catalog = gz2_dataset.catalog
    gz2_label_cols = gz2_dataset.label_cols

### Download Notes

Datasets are downloaded like:

- {root}
    - images
        - subfolder (except GZ2)
            - image.jpg
    - {catalog_name(s)}.parquet