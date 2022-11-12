from typing import List
import logging

import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from PIL import Image
import simplejpeg



# https://pytorch.org/tutorials/beginner/basics/data_tutorial.html
class GalaxyDataset(Dataset):
    def __init__(self, catalog: pd.DataFrame, label_cols: List, transform=None, target_transform=None):
        """
        Create custom PyTorch Dataset using catalog of galaxy images

        Catalog and images should already be downloaded.
        Use methods under galaxy_datasets.prepared_datasets e.g. prepared_datasets.gz2()
        
        Note that if you want the canonical datasets (i.e. using the standard train/test split)
        you can use the galaxy_datasets.pytorch.datasets instead of this class e.g. 

        ```
        datasets.GZ2Dataset(
            root=os.path.join(some_dir, 'gz2'),
            download=True
        )
        ```
        
        Reads a catalog of jpeg galaxy images
        if transform is from albumentations, datamodule should know about the transform including 
        catalog should be split already
        should have correct image locations under file_loc

        Args:
            catalog (pd.DataFrame): with images under 'file_loc' and labels under label_cols
            label_cols (List): columns in catalog with labels
            transform (callable, optional): See Pytorch Datasets. Defaults to None.
            target_transform (callable, optional): See Pytorch Datasets. Defaults to None.
        """
        # internal catalog has id_str as index for query speed
        # (dataset.catalog still returns the int-indexed catalog via the catalog method below)
        # self._catalog = catalog.copy().set_index('id_str', verify_integrity=True)
        self.catalog = catalog
        
        self.label_cols = label_cols
        self.transform = transform
        self.target_transform = target_transform


    def __len__(self) -> int:
        return len(self.catalog)


    def __getitem__(self, idx: int):
        #the index is id_str so can use that for quick search on 1M+ catalo
        # galaxy = self._catalog.loc[idx]
        galaxy = self.catalog.iloc[idx]
        # option A
        # img_path = galaxy['file_loc']
        # image = read_image(img_path) # PIL under the hood: Returns CHW Tensor.
        # option B - tiny bit faster when CPU-limited
        with open(galaxy['file_loc'], 'rb') as f:
            # image = torch.from_numpy(decode_jpeg(f.read()).transpose(2, 0, 1))  # CHW tensor
            try:
                image = Image.fromarray(decode_jpeg(f.read()))  # HWC PIL image via simplejpeg
            except Exception as e:
                logging.critical('Cannot load {}'.format(galaxy['file_loc']))
                raise e
        label = get_galaxy_label(galaxy, self.label_cols)

        # logging.info((image.shape, torch.max(image), image.dtype, label))  # always 0-255 uint8

        if self.transform:
            # a CHW tensor, which torchvision wants. May change to PIL image.
            image = self.transform(image)

        if self.target_transform:
            label = self.target_transform(label)

        # logging.info((image.shape, torch.max(image), image.dtype, label))  #  should be 0-1 float
        return image, label


def load_encoded_jpeg(loc: str):
    with open(loc, "rb") as f:
        return f.read()  # bytes, not yet decoded


def decode_jpeg(encoded_bytes):
    return simplejpeg.decode_jpeg(encoded_bytes, fastdct=True, fastupsample=True)


def get_galaxy_label(galaxy: pd.Series, label_cols: List) -> np.ndarray:
    # no longer casts to int64, user now responsible in df. If dtype is mixed, will try to infer with infer_objects
    return galaxy[label_cols].infer_objects().values.squeeze()  # squeeze for if there's one label_col
