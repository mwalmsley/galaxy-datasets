import pytest

import os
import numpy as np
from pytorch_galaxy_datasets.prepared_datasets import candels, dr5, gz2, legs, rings, tidal

# first as smallest
def test_download_tidal_dataset(tmp_path):
    dataset = tidal.TidalDataset(
        train=True,
        root=tmp_path,
        download=True
    )

    for image, label in dataset:
        print(np.array(image).shape, label)
        break


def test_download_candels_dataset(tmp_path):
    dataset = candels.CandelsDataset(
        root=tmp_path,
        download=True
    )
    for image, label in dataset:
        print(np.array(image).shape, label)
        break


def test_download_dr5_dataset(tmp_path):
    dataset = dr5.DecalsDR5Dataset(
        root=tmp_path,
        download=True
    )
    for image, label in dataset:
        print(np.array(image).shape, label)
        break


def test_download_gz2_dataset(tmp_path):
    dataset = gz2.GZ2Dataset(
        root=tmp_path,
        download=True
    )
    for image, label in dataset:
        print(np.array(image).shape, label)
        break


def test_download_rings_dataset(tmp_path):

    dataset = rings.RingsDataset(
        # root='/nvme1/scratch/walml/repos/pytorch-galaxy-datasets/roots/rings_root/temp',
        root=tmp_path,
        download=True
    )
    for image, label in dataset:
        print(np.array(image).shape, label)
        break


@pytest.mark.skipif(not os.path.isdir('/share/nas2/walml'), reason="Data only exists on Galahad")
def test_download_legs_dataset(tmp_path):
    dataset = legs.LegsDataset(
        root=tmp_path,
        download=True,
        train='labelled'
    )
    for image, label in dataset:
        print(np.array(image).shape, label)
        break

