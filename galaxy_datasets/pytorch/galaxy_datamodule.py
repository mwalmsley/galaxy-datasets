import logging
import os
from typing import Optional, Union, List

import numpy as np
import pandas as pd
import lightning as L
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from galaxy_datasets.pytorch import dataset_utils
from galaxy_datasets.transforms import get_galaxy_transform, default_view_config, minimal_view_config


# import torch
from torch.utils.data import DataLoader
import datasets as hf_datasets

from galaxy_datasets.pytorch import galaxy_dataset
# for type checking
from torchvision.transforms.v2 import Compose

# https://pytorch-lightning.readthedocs.io/en/stable/extensions/datamodules.html
class CatalogDataModule(L.LightningDataModule):
    # takes generic catalogs (which are already downloaded and happy),
    # splits if needed, and creates generic datasets->dataloaders etc
    # easy to make dataset-specific default transforms if desired
    def __init__(
        self,
        label_cols: Union[List, None],
        train_transform: Optional[Compose] = None,
        test_transform: Optional[Compose] = None,
        # provide full catalog for automatic split, or...
        catalog=None,
        train_fraction=0.7,
        val_fraction=0.1,
        test_fraction=0.2,
        # provide train/val/test catalogs for your own previous split
        train_catalog: Optional[pd.DataFrame] = None,
        val_catalog: Optional[pd.DataFrame] = None,
        test_catalog: Optional[pd.DataFrame] = None,
        predict_catalog: Optional[pd.DataFrame] = None,

        # hardware params
        batch_size=256,  # careful - will affect final performance
        use_memory=False,  # deprecated
        num_workers=4,
        prefetch_factor=4,
        seed=42,
    ):
        super().__init__()

        if catalog is not None:  # catalog provided, should not also provide explicit split catalogs
            assert train_catalog is None
            assert val_catalog is None
            assert test_catalog is None
        else:  # catalog not provided, must provide explicit split catalogs - at least one
            assert (
                (train_catalog is not None)
                or (val_catalog is not None)
                or (test_catalog is not None)
                or (predict_catalog is not None)
            )
            # see setup() for how having only some explicit catalogs is handled

        self.label_cols = label_cols

        if train_transform is None:
            logging.warning("No train transform requested, using default galaxy transforms")
            self.train_transform = get_galaxy_transform(default_view_config())
        else:
            self.train_transform = train_transform
            
        if test_transform is None:
            logging.warning("No test transform requested, using minimal galaxy transforms")
            self.test_transform = get_galaxy_transform(minimal_view_config())
        else:
            self.test_transform = test_transform

        self.catalog = catalog
        self.train_catalog = train_catalog
        self.val_catalog = val_catalog
        self.test_catalog = test_catalog
        self.predict_catalog = predict_catalog

        self.batch_size = batch_size

        self.use_memory = use_memory
        if self.use_memory:
            raise NotImplementedError

        self.num_workers = num_workers
        self.seed = seed

        assert np.isclose(train_fraction + val_fraction + test_fraction, 1.0)
        self.train_fraction = train_fraction
        self.val_fraction = val_fraction
        self.test_fraction = test_fraction

        if self.num_workers == 0:
            logging.warning(
                "num_workers=0, setting prefetch=None and timeout=0 as no multiprocessing"
            )
            self.prefetch_factor = None
            self.dataloader_timeout = 0
        else:
            self.prefetch_factor = prefetch_factor
            self.dataloader_timeout = 600  # seconds aka 10 mins

        logging.info("Num workers: {}".format(self.num_workers))
        logging.info("Prefetch factor: {}".format(self.prefetch_factor))

        self.transform_with_torchvision()

    def transform_with_torchvision(self):
        # easy to accidentally pass the cfg objects
        assert isinstance(self.train_transform, Compose), type(self.train_transform)
        assert isinstance(self.test_transform, Compose), type(self.test_transform)

    # only called on main process
    def prepare_data(self):
        pass  # could include some basic checks

    # called on every gpu

    def setup(self, stage: Optional[str] = None):

        self.specify_catalogs(stage)

        # Assign train/val datasets for use in dataloaders
        # assumes dataset_class has these standard args
        if stage == "fit" or stage is None:
            self.train_dataset = galaxy_dataset.CatalogDataset(
                catalog=self.train_catalog,
                label_cols=self.label_cols,
                transform=self.train_transform,
            )
            self.val_dataset = galaxy_dataset.CatalogDataset(
                catalog=self.val_catalog,
                label_cols=self.label_cols,
                transform=self.test_transform,
            )

        # Assign test dataset for use in dataloader(s)
        if stage == "test" or stage is None:
            self.test_dataset = galaxy_dataset.CatalogDataset(
                catalog=self.test_catalog,
                label_cols=self.label_cols,
                transform=self.test_transform,
            )

        if (
            stage == "predict"
        ):  # not set up by default with stage=None, only if explicitly requested
            if self.predict_catalog is None:
                raise ValueError(
                    "Attempting to predict, but GalaxyDataModule was init without a predict_catalog arg. init with GalaxyDataModule(predict_catalog=some_catalog, ...)"
                )
            self.predict_dataset = galaxy_dataset.CatalogDataset(
                catalog=self.predict_catalog,
                label_cols=self.label_cols,
                transform=self.test_transform,
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
            prefetch_factor=self.prefetch_factor,
            timeout=self.dataloader_timeout,
            drop_last=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
            prefetch_factor=self.prefetch_factor,
            timeout=self.dataloader_timeout,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
            prefetch_factor=self.prefetch_factor,
            timeout=self.dataloader_timeout,
        )

    def predict_dataloader(self):
        return DataLoader(
            self.predict_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
            prefetch_factor=self.prefetch_factor,
            timeout=self.dataloader_timeout,
        )

    def specify_catalogs(self, stage):
        if self.catalog is not None:
            # will split the catalog into train, val, test here
            self.train_catalog, hidden_catalog = train_test_split(
                self.catalog, train_size=self.train_fraction, random_state=self.seed
            )
            self.val_catalog, self.test_catalog = train_test_split(
                hidden_catalog,
                train_size=self.val_fraction / (self.val_fraction + self.test_fraction),
                random_state=self.seed,
            )
            del hidden_catalog
        else:
            # assume you have passed pre-split catalogs
            # (maybe not all, e.g. only a test catalog, or only train/val catalogs)
            if stage == "predict":
                assert self.predict_catalog is not None
            elif stage == "test":
                # only need test
                assert self.test_catalog is not None
            elif stage == "fit":
                # only need train and val
                assert self.train_catalog is not None
                assert self.val_catalog is not None
            else:
                # need all three (predict is still optional)
                assert self.train_catalog is not None
                assert self.val_catalog is not None
                assert self.test_catalog is not None
            # (could write this shorter but this is clearest)

# moved from gz-evo to here
# https://pytorch-lightning.readthedocs.io/en/stable/extensions/datamodules.html
class HuggingFaceDataModule(CatalogDataModule):
    def __init__(
        self,
        dataset_dict: hf_datasets.DatasetDict,  # must have train and test keys
        train_transform,
        test_transform,
        # target_transform=None,
        # hardware params
        batch_size=256,
        num_workers=4,
        prefetch_factor=4,
        seed=42,
        iterable=False  # whether to use IterableDataset (faster, no indexed access)
        # dataset_kwargs={}
    ):
        # super().__init__()
        L.LightningDataModule.__init__(self)

        logging.info("Initializing HuggingFaceDataModule")

        self.batch_size = batch_size

        self.num_workers = num_workers
        self.seed = seed

        self.dataset_dict = dataset_dict

        self.train_transform = train_transform
        self.test_transform = test_transform

        # self.target_transform = target_transform

        # self.dataset_kwargs = dataset_kwargs

        self.iterable = iterable

        if self.num_workers == 0:
            logging.warning(
                "num_workers=0, setting prefetch=None and timeout=0 as no multiprocessing"
            )
            self.prefetch_factor = None
            self.dataloader_timeout = 0
        else:
            self.prefetch_factor = prefetch_factor
            self.dataloader_timeout = 600  # seconds aka 10 mins

        logging.info("Num workers: {}".format(self.num_workers))
        logging.info("Prefetch factor: {}".format(self.prefetch_factor))

        # self.prepare_data_per_node = False  # run prepare_data below only on master node (and one process)

    # only called on main process
    def prepare_data(self):
        pass

    # torchvision acts on image key but HF dataset returns dicts
    def train_transform_wrapped(self, example: dict):
        # REPLACES set_transform('torch') so we also need to make torch tensors
        # https://huggingface.co/docs/datasets/v3.6.0/en/package_reference/main_classes#datasets.Dataset.with_transform
        # best with pil_to_tensor=True
        example['image'] = self.train_transform(example['image'])
        return example
    def test_transform_wrapped(self, example: dict):
        example['image'] = self.test_transform(example['image'])
        return example
    # .map sends example as dict
    # .set_transform sends example as dict of lists, i.e. a batched dict
    # torch collate func will handle the final dict-of-lists-to-tensor, but image transforms only get applied to the first img
    def train_transform_wrapped_batch(self, examples: dict):
        # assert len(examples['image']) > 1
        examples['image'] = [self.train_transform(im) for im in examples['image']]
        # maybe it's batchwise compatible, but not sure?
        # examples['image'] = self.train_transform(examples['image'])  # stack to tensor
        return examples
    def test_transform_wrapped_batch(self, examples: dict):
        examples['image'] = [self.test_transform(im) for im in examples['image']]
        # examples['image'] = self.test_transform(examples['image'])  # stack to tensor
        return examples

    # called on every gpu
    def setup(self, stage: Optional[str] = None):

        if 'validation' not in self.dataset_dict.keys():
            # if no validation split, add it
            logging.info('No validation split found, adding one')
            self.dataset_dict = dataset_utils.add_validation_split(self.dataset_dict, seed=self.seed, num_workers=self.num_workers)


        if stage == "fit" or stage is None:

            if self.iterable:
                # convert to iterable datasets
                logging.info('Converting to iterable datasets')
                # these have been split above, is really train and val
                train_dataset_hf = self.dataset_dict["train"].to_iterable_dataset(num_shards=64)
                val_dataset_hf = self.dataset_dict["validation"].to_iterable_dataset(num_shards=64)

                # apply transforms with map
                # map passes each example through the transform function as a dict
                # (while with_transform sends a list...)
                train_dataset_hf = train_dataset_hf.map(self.train_transform_wrapped)
                val_dataset_hf = val_dataset_hf.map(self.test_transform_wrapped)
                # https://huggingface.co/docs/datasets/en/image_process
                # for dataset, map is cached and intended for "do once" transforms
                # with_format('torch') is (probably) a map
                # set_transform is intended for "on-the-fly" transforms and so is not cached (less disk space, faster)
                # for iterabledataset, map is applied on-the-fly (on every yield) and not cached
                # so there's no need for set_transform: everything is a non-cached map
    
            else:  # leave as not iterable, fast reads, but list comprehension for transforms
                train_dataset_hf = self.dataset_dict['train']
                val_dataset_hf = self.dataset_dict['validation']

                # set transforms to use on-the-fly
                # with transform only works with dataset, not iterabledataset
                train_dataset_hf = train_dataset_hf.with_transform(self.train_transform_wrapped_batch)
                val_dataset_hf = val_dataset_hf.with_transform(self.train_transform_wrapped_batch)
                # these act individually, dataloader will handle batching afterwards

            self.train_dataset = train_dataset_hf
            self.val_dataset = val_dataset_hf

        # Assign test dataset for use in dataloader(s)
        if stage == "test" or stage is None:
            test_dataset_hf = self.dataset_dict['test']
            # not shuffled, so no need to flatten indices
            # never iterable, for now
            test_dataset_hf = test_dataset_hf.with_transform(self.test_transform_wrapped_batch)
            self.test_dataset = test_dataset_hf




    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,  # type: ignore
            batch_size=self.batch_size,
            shuffle=False,  # assume preshuffled
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
            prefetch_factor=self.prefetch_factor,
            timeout=self.dataloader_timeout,
            drop_last=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,  # type: ignore
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
            prefetch_factor=self.prefetch_factor,
            timeout=self.dataloader_timeout,
            drop_last=True
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,  # type: ignore
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
            prefetch_factor=self.prefetch_factor,
            timeout=self.dataloader_timeout,
            drop_last=False
        )

    # not used within lightning, but helpful
    def get_predict_dataloader(self, split: str):

        # never iterable and always with test transform
        predict_dataset = self.dataset_dict[split].with_transform(self.test_transform_wrapped_batch)

        return DataLoader(
            predict_dataset,  # type: ignore
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
            prefetch_factor=self.prefetch_factor,
            timeout=self.dataloader_timeout,
            drop_last=False
        )