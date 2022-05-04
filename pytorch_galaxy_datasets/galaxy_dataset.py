import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import simplejpeg


import torch
import os
import logging

import numpy as np
import pandas as pd


# https://pytorch.org/tutorials/beginner/basics/data_tutorial.html
class GalaxyDataset(Dataset):
    def __init__(self, catalog: pd.DataFrame, label_cols, transform=None, target_transform=None):
        # downloaded dataset where everything is all prepared, created by GalaxyDatasetFactory
        # reads a catalog of jpeg images, assumed of galaxies
        # if transform is from albumentations, datamodule should know about the transform including 
        # catalog should be split already
        # should have correct image locations under file_loc
        self.catalog = catalog
        self.label_cols = label_cols
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.catalog)

    def __getitem__(self, idx):
        galaxy = self.catalog.iloc[idx]
        # option A
        # img_path = galaxy['file_loc']
        # image = read_image(img_path) # PIL under the hood: Returns CHW Tensor.
        # option B - tiny bit faster when CPU-limited
        with open(galaxy['file_loc'], 'rb') as f:
            # image = torch.from_numpy(decode_jpeg(f.read()).transpose(2, 0, 1))  # CHW tensor
            image = Image.fromarray(decode_jpeg(f.read()))  # HWC PIL image via simplejpeg
        label = get_galaxy_label(galaxy, self.label_cols)

        # logging.info((image.shape, torch.max(image), image.dtype, label))  # always 0-255 uint8

        if self.transform:
            # a CHW tensor, which torchvision wants. May change to PIL image.
            image = self.transform(image)

        if self.target_transform:
            label = self.target_transform(label)

        # logging.info((image.shape, torch.max(image), image.dtype, label))  #  should be 0-1 float
        return image, label


def load_encoded_jpeg(loc):
    with open(loc, "rb") as f:
        return f.read()  # bytes, not yet decoded


def decode_jpeg(encoded_bytes):
    return simplejpeg.decode_jpeg(encoded_bytes, fastdct=True, fastupsample=True)


def get_galaxy_label(galaxy, label_cols):
    # TODO currently assumes int i.e. num of votes or class label
    return galaxy[label_cols].values.astype(np.int64).squeeze()  # squeeze for if there's one label_col




# # https://pytorch.org/vision/stable/_modules/torchvision/datasets/mnist.html for download process inspiration
# class GalaxyDatasetFactory():

#     def __init__(self, root, catalog=None, label_cols=None,  transform=None, target_transform=None) -> None:

#         # can use target_transform to turn counts into regression or even classification
#         # will need another step to drop rows, in DataModule probably

#         self.root = root

#         catalog, label_cols = self.adjust_catalog_and_labels(root, catalog, label_cols)

#         super().__init__(catalog=catalog, label_cols=label_cols,
#                          transform=transform, target_transform=target_transform)


#     def adjust_catalog_and_labels(self, root, catalog, label_cols):
#         if not self._check_exists():
#             raise RuntimeError(
#                 "Dataset not found. You can use download=True to download it")

#         if catalog is None:
#             logging.info('Loading dataset with default (unsplit) catalog')
#             catalog = pd.read_parquet(os.path.join(root, self.default_catalog_loc))
#             catalog['file_loc'] = catalog['filename'].apply(
#                 lambda x: os.path.join(self.image_dir, x))
#         else:
#             logging.info(
#                 'Overriding default catalog with user-provided catalog (length {})'.format(len(catalog)))
#             assert isinstance(catalog, pd.DataFrame)
#             # will always check label_cols as well, below
#             assert 'file_loc' in catalog.columns.values

#         if label_cols is None:
#             logging.info('Loading dataset with default label columns')
#             label_cols = self.default_label_cols
#         else:
#             logging.info('User provided GZ2 dataset with custom label cols {}'.format(label_cols))
#                 # label_cols_present = [col in catalog.columns.values for col in label_cols]
#             missing_label_cols = set(label_cols) - set(catalog.columns.values)
#             if len(missing_label_cols) > 0:
#                 raise KeyError(f'User asked for label columns not present in catalog:\n{missing_label_cols} not in \n{catalog.columns.values}')
#         return catalog,label_cols
