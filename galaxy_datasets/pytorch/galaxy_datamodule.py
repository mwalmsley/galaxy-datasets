
from typing import Optional, Union
import logging
from functools import partial

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
# import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from torchvision.transforms import v2 as T

from galaxy_datasets.pytorch import galaxy_dataset
from galaxy_datasets.transforms import default_transforms


# https://pytorch-lightning.readthedocs.io/en/stable/extensions/datamodules.html
class GalaxyDataModule(pl.LightningDataModule):
    # takes generic catalogs (which are already downloaded and happy),
    # splits if needed, and creates generic datasets->dataloaders etc
    # easy to make dataset-specific default transforms if desired
    def __init__(
        self,
        label_cols,
        # provide full catalog for automatic split, or...
        catalog=None,
        train_fraction=0.7,
        val_fraction=0.1,
        test_fraction=0.2,
        # provide train/val/test catalogs for your own previous split
        train_catalog: Optional[pd.DataFrame]=None,
        val_catalog: Optional[pd.DataFrame]=None,
        test_catalog: Optional[pd.DataFrame]=None,
        predict_catalog: Optional[pd.DataFrame]=None,
        # augmentation params (sensible supervised defaults)
        greyscale=True,
        # album=False,  # now True always
        crop_scale_bounds=(0.7, 0.8),
        crop_ratio_bounds=(0.9, 1.1),
        resize_after_crop=224,
        custom_albumentation_transform=None,  # will override the settings above. If tuple, assume (train, test) transforms
        custom_torchvision_transform=None,  # similarly
        # hardware params
        batch_size=256,  # careful - will affect final performance
        use_memory=False,  # deprecated
        num_workers=4,
        prefetch_factor=4,
        seed=42
    ):
        super().__init__()

        if catalog is not None:  # catalog provided, should not also provide explicit split catalogs
            assert train_catalog is None
            assert val_catalog is None
            assert test_catalog is None
        else:  # catalog not provided, must provide explicit split catalogs - at least one
            assert (train_catalog is not None) or (val_catalog is not None) or (test_catalog is not None) or (predict_catalog is not None)
            # see setup() for how having only some explicit catalogs is handled
            

        self.label_cols = label_cols

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

        assert np.isclose(train_fraction + val_fraction + test_fraction, 1.)
        self.train_fraction = train_fraction
        self.val_fraction = val_fraction
        self.test_fraction = test_fraction

        if self.num_workers == 0:
            logging.warning('num_workers=0, setting prefetch=None and timeout=0 as no multiprocessing')
            self.prefetch_factor = None
            self.dataloader_timeout = 0
        else:
            self.prefetch_factor = prefetch_factor
            self.dataloader_timeout = 600  # seconds aka 10 mins

        logging.info('Num workers: {}'.format(self.num_workers))
        logging.info('Prefetch factor: {}'.format(self.prefetch_factor))


        self.custom_albumentation_transform = custom_albumentation_transform
        self.custom_torchvision_transform = custom_torchvision_transform
        self.resize_after_crop = resize_after_crop
        self.crop_scale_bounds = crop_scale_bounds
        self.crop_ratio_bounds = crop_ratio_bounds
        self.greyscale = greyscale

        if custom_torchvision_transform is not None:
            logging.info('Using custom torchvision transform for augmentations')
            self.transform_with_torchvision()
        else:
            logging.info('Using albumentations transform for augmentations')
            self.transform_with_albumentations()

    def transform_with_torchvision(self):
        # TODO
        # galaxy_datasets loads PIL images, likely need PILToTensor() in any custom torchvision transform. review.
        if isinstance(self.custom_torchvision_transform, tuple):
            logging.info('Using different torchvision transforms for train and test')
            assert len(self.custom_torchvision_transform) == 2
            self.train_transform = self.custom_torchvision_transform[0]
            self.test_transform = self.custom_torchvision_transform[1]
        else:
            self.train_transform = self.custom_torchvision_transform
            self.test_transform = self.custom_torchvision_transform

    def transform_with_albumentations(self):
        import albumentations as A
        if self.custom_albumentation_transform is not None:
            if isinstance(self.custom_albumentation_transform, tuple):
                logging.info('Using different albumentations transforms for train and test')
                assert len(self.custom_albumentation_transform) == 2
                assert isinstance(self.custom_albumentation_transform[0], A.Compose)
                self.train_transform = partial(do_transform, transforms_to_apply=self.custom_albumentation_transform[0])
                self.test_transform = partial(do_transform, transforms_to_apply=self.custom_albumentation_transform[1])
            else:
                logging.info('Using the same custom albumentations transforms for train and test')
                self.train_transform = partial(do_transform, transforms_to_apply=self.custom_albumentation_transform)
                self.test_transform = partial(do_transform, transforms_to_apply=self.custom_albumentation_transform)
        else:
            logging.info('Using basic albumentations transforms for augmentations')
            # gives a transforms = Compose() object
            transforms_to_apply = default_transforms(
                crop_scale_bounds=self.crop_scale_bounds,
                crop_ratio_bounds=self.crop_ratio_bounds,
                resize_after_crop=self.resize_after_crop,
                pytorch_greyscale=self.greyscale
            )
            self.train_transform = partial(do_transform, transforms_to_apply=transforms_to_apply)
            self.test_transform = partial(do_transform, transforms_to_apply=transforms_to_apply)

        

    # only called on main process
    def prepare_data(self):
        pass   # could include some basic checks

    # called on every gpu

    def setup(self, stage: Optional[str] = None):

        self.specify_catalogs(stage)

        # Assign train/val datasets for use in dataloaders
        # assumes dataset_class has these standard args
        if stage == "fit" or stage is None:
            self.train_dataset = galaxy_dataset.GalaxyDataset(
                catalog=self.train_catalog, label_cols=self.label_cols, transform=self.train_transform
            )
            self.val_dataset = galaxy_dataset.GalaxyDataset(
                catalog=self.val_catalog, label_cols=self.label_cols, transform=self.test_transform
            )

        # Assign test dataset for use in dataloader(s)
        if stage == "test" or stage is None:
            self.test_dataset = galaxy_dataset.GalaxyDataset(
                catalog=self.test_catalog, label_cols=self.label_cols, transform=self.test_transform
            )

        if stage == 'predict':  # not set up by default with stage=None, only if explicitly requested
            if self.predict_catalog is None:
                raise ValueError('Attempting to predict, but GalaxyDataModule was init without a predict_catalog arg. init with GalaxyDataModule(predict_catalog=some_catalog, ...)')
            self.predict_dataset = galaxy_dataset.GalaxyDataset(
                catalog=self.predict_catalog, label_cols=self.label_cols, transform=self.test_transform
            )

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, pin_memory=True, persistent_workers=self.num_workers > 0, prefetch_factor=self.prefetch_factor, timeout=self.dataloader_timeout, drop_last=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=True, persistent_workers=self.num_workers > 0, prefetch_factor=self.prefetch_factor, timeout=self.dataloader_timeout)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=True, persistent_workers=self.num_workers > 0, prefetch_factor=self.prefetch_factor, timeout=self.dataloader_timeout)

    def predict_dataloader(self):
        return DataLoader(self.predict_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=True, persistent_workers=self.num_workers > 0, prefetch_factor=self.prefetch_factor, timeout=self.dataloader_timeout)

    def specify_catalogs(self, stage):
        if self.catalog is not None:
            # will split the catalog into train, val, test here
            self.train_catalog, hidden_catalog = train_test_split(
                self.catalog, train_size=self.train_fraction, random_state=self.seed
            )
            self.val_catalog, self.test_catalog = train_test_split(
                hidden_catalog, train_size=self.val_fraction/(self.val_fraction + self.test_fraction), random_state=self.seed
            )
            del hidden_catalog
        else:
            # assume you have passed pre-split catalogs
            # (maybe not all, e.g. only a test catalog, or only train/val catalogs)
            if stage == 'predict':
                assert self.predict_catalog is not None
            elif stage == 'test':
                # only need test
                assert self.test_catalog is not None
            elif stage == 'fit':
                # only need train and val
                assert self.train_catalog is not None
                assert self.val_catalog is not None
            else:
                # need all three (predict is still optional)
                assert self.train_catalog is not None
                assert self.val_catalog is not None
                assert self.test_catalog is not None
            # (could write this shorter but this is clearest)
            

def do_transform(img, transforms_to_apply):
    # albumentations expects np array, and returns dict keyed by "image"
    # transpose changes from BHWC (numpy/TF style) to BCHW (torch style) 
    # cannot use a lambda or define here because must be pickleable for multi-gpu
    return np.transpose(transforms_to_apply(image=np.array(img))["image"], axes=[2, 0, 1]).astype(np.float32)



# deprecated for albumentations
# def default_torchvision_transforms(greyscale, resize_size, crop_scale_bounds, crop_ratio_bounds):
#     # refactored out for use elsewhere, if need exactly these transforms
#     # assume input is 0-255 uint8 tensor

#     # automatically normalises from 0-255 int to 0-1 float
#     transforms_to_apply = [transforms.ToTensor()]  # dataset gives PIL image currently

#     if greyscale:
#         # transforms.Grayscale() adds perceptual weighting to rgb channels
#         transforms_to_apply += [GrayscaleUnweighted()]

#     transforms_to_apply += [
#         transforms.RandomResizedCrop(
#             size=resize_size,  # assumed square
#             scale=crop_scale_bounds,  # crop factor
#             ratio=crop_ratio_bounds,  # crop aspect ratio
#             interpolation=transforms.InterpolationMode.BILINEAR),  # new aspect ratio
#         transforms.RandomHorizontalFlip(),
#         transforms.RandomRotation(
#             degrees=180., interpolation=transforms.InterpolationMode.BILINEAR)
#     ]
    
#     return transforms_to_apply

# torchvision
# class GrayscaleUnweighted(torch.nn.Module):

#     def __init__(self, num_output_channels=1):
#         super().__init__()
#         self.num_output_channels = num_output_channels

#     def forward(self, img):
#         """
#         PyTorch (and tensorflow) does greyscale conversion as a *weighted* mean by default (as colours have different perceptual brightnesses).
#         Here, do a simple mean.
#         Args:
#             img (Tensor): Image to be converted to grayscale.

#         Returns:
#             Tensor: Grayscaled image.
#         """
#         # https://pytorch.org/docs/stable/generated/torch.mean.html
#         return img.mean(dim=-3, keepdim=True)  # (..., C, H, W) convention

#     def __repr__(self):
#         return self.__class__.__name__ + '(num_output_channels={0})'.format(self.num_output_channels)


