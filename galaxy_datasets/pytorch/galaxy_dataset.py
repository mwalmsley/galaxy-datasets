import logging
from typing import List

import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
import torch
import datasets as hf_datasets  # HuggingFace datasets

# not a strict requirement unless loading fits
try:
    from astropy.io import fits
except ImportError:
    pass


# https://pytorch.org/tutorials/beginner/basics/data_tutorial.html
class GalaxyDataset(Dataset):
    def __init__(
        self,
        catalog: pd.DataFrame,
        label_cols=None,
        transform=None,
        target_transform=None,
    ):
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
        # the index is id_str so can use that for quick search on 1M+ catalo
        # galaxy = self._catalog.loc[idx]
        galaxy = self.catalog.iloc[idx]

        # load the image into memory
        image_loc = galaxy["file_loc"]
        try:
            image = load_img_file(image_loc)
            # HWC PIL image
            # logging.info((image.shape, torch.max(image), image.dtype, label))  # always 0-255 uint8
        except Exception as e:
            logging.critical("Cannot load {}".format(image_loc))
            raise e

        if self.transform:
            image = self.transform(image)

        if self.label_cols is None:
            return image
        else:
            # load the labels. If no self.label_cols, will
            label = get_galaxy_label(galaxy, self.label_cols)

            if self.target_transform:
                label = self.target_transform(label)

            return image, label


# https://huggingface.co/docs/datasets/en/use_with_pytorch
# https://huggingface.co/docs/datasets/en/image_process
class HF_GalaxyDataset(Dataset):
    def __init__(
        self,
        dataset: hf_datasets.Dataset,  # HF Dataset
        label_cols=['label'],
        transform=None,
        target_transform=None,
    ):
        """
        Create custom PyTorch Dataset using HuggingFace dataset


        Args:
            name (str): Name of HF dataset
            transform (callable, optional): See Pytorch Datasets. Defaults to None.
            target_transform (callable, optional): See Pytorch Datasets. Defaults to None.
        """
        self.dataset = dataset
        self.label_cols = label_cols
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int):
        example:dict = self.dataset[idx]
        
        if self.transform:
             example['image'] = self.transform(example['image'])

        if self.target_transform:
            example = self.target_transform(example)  # slightly generalised: target_transform to expect and yield example, changing the targets (labels)

        return (example['image'], torch.stack([example[label] for label in self.label_cols]).squeeze())
        # ultimately I would prefer to migrate to 
        # return example  # dict like {'image': image, 'label_a': label_a, ...}



def load_img_file(loc):
    # wrapper around specific loaders below
    # could be more performant with a dict of {format:loader} but doubt significant
    if loc.endswith("png"):
        return load_png_file(loc)
    elif loc.endswith("jpg"):
        return load_jpg_file(loc)
    elif loc.endswith("jpeg"):
        return load_jpg_file(loc)
    elif loc.endswith("fits"):
        return load_fits_file(
            loc
        )  # careful, these often need a transform to have reasonable dynamic range
    else:
        raise ValueError(
            "File format of {} not recognised - should be jpeg|jpg (preferred) or png".format(loc)
        )


def load_jpg_file(loc):
    im = Image.open(loc, mode="r")  # HWC
    im.load()  # avoid lazy open
    return im
    # below works, but deprecated due to simplejpeg install issue on my M1 mac
    # let's just keep dependencies simple...
    # with open(loc, 'rb') as f:
    # return Image.fromarray(decode_jpeg(f.read()))  # HWC PIL image via simplejpeg


# def decode_jpeg(encoded_bytes):
#     return simplejpeg.decode_jpeg(encoded_bytes, fastdct=True, fastupsample=True)


def load_png_file(loc):
    # TODO now duplicate with the above
    im = Image.open(loc, mode="r")  # HWC
    im.load()  # avoid lazy open
    return im


def load_fits_file(loc):
    x = fits.open(loc)[0].data.astype(np.float32)
    # assumes single channel - add channel dimension
    return np.expand_dims(x, axis=2)  # HWC


def get_galaxy_label(galaxy: pd.Series, label_cols: List) -> np.ndarray:
    # pytorch 1.12 is happy with float32 for both dirichlet and cross-entropy
    return (
        galaxy[label_cols].astype(np.float32).values.squeeze()
    )  # squeeze for if there's one label_col


if __name__ == "__main__":

    # lazy test/example
    import glob

    mixed_file_paths = glob.glob("tests/test_data/png_jpg_mix/*")
    assert len(mixed_file_paths) > 0
    data = {
        "file_loc": mixed_file_paths,
        "id_str": [str(x) for x in np.arange(len(mixed_file_paths))],
    }
    df = pd.DataFrame(data)

    dataset = GalaxyDataset(catalog=df)
    for im in dataset:
        im = np.array(im)  # returns PIL.Image, if not given label_cols or transform
        print(im.shape)
        print(im.mean(), im.min(), im.max())
