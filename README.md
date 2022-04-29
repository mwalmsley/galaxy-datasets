# pytorch-galaxy-datasets
PyTorch Datasets and PyTorch Lightning Datamodules for loading images and labels from Galaxy Zoo citizen science campaigns.

| Name      | Class | Published | Downloadable
| ----------- | ----------- | --- | ---- |
| Galaxy Zoo 2 | GZ2 | &#x2611; | &#x2611;
| GZ DECaLS GZD-5   | DecalsDR5 | &#x2611; | &#x2611; |
| Galaxy Zoo Rings | Rings | &#x2612; | &#x2611; |
| GZ Legacy Survey  | Legs | &#x2612; | &#x2612; |

Any datasets marked as downloadable are only downloadable internally until published.

If a dataset is published but not marked as downloadable (none currently), it means I haven't yet got around to making the download automatic. You can still download it via the paper instructions.

You may also be interested in [Galaxy MNIST](https://github.com/mwalmsley/galaxy_mnist) as a simple dataset for teaching/debugging.

For each dataset, you must cite/acknowledge the GZ data release paper and the original telescope survey from which the images were derived.

### Installation

For local development (e.g. adding a new dataset), you can install this by cloning from github, then running `pip install -e .` in the cloned repo root. 

It will also be installed by default as a dependency of `zoobot` if you specify the pytorch version of `zoobot` - but this is slightly trickier if you'd like to make changes as it'll be installed under your `sitepackages`.

### Usage

You can load each prepared dataset as a pytorch Dataset like so:

    from pytorch_galaxy_datasets.prepared_datasets import GZ2Dataset

    gz2_dataset = GZ2Dataset(
        data_dir='/nvme1/scratch/walml/repos/pytorch-galaxy-datasets/tests/gz2_root',
        download=False
    )
    image, label = gz2_dataset[0]
    plt.imshow(image)

You will probably want to customise the dataset, selecting a subset of galaxies or labels. Do this with the `{dataset}_setup()` methods.

    from pytorch_galaxy_datasets.prepared_datasets import gz2_setup

    catalog, label_cols = gz2_setup(
        data_dir='/nvme1/scratch/walml/repos/pytorch-galaxy-datasets/tests/gz2_root',
        download=False
    )
    adjusted_catalog = gz2_catalog.sample(1000)

You can then customise the catalog and labels before creating a generic GalaxyDataModule, which has default transforms for supervised learning.

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

- {data_dir}
    - images
        - subfolder (except GZ2)
            - image.jpg
    - {catalog_name(s)}.parquet