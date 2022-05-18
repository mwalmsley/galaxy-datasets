import logging
import os

import pandas as pd
from pytorch_galaxy_datasets import galaxy_datamodule, galaxy_dataset, download_utils
from pytorch_galaxy_datasets.prepared_datasets import internal_urls

# TODO could eventually refactor this out of Zoobot as well
from zoobot.shared import label_metadata


class LegsDataset(galaxy_dataset.GalaxyDataset):
    
    # based on https://pytorch.org/vision/stable/generated/torchvision.datasets.STL10.html
    def __init__(self, root=None, split='train', download=False, transform=None, target_transform=None, train=None):
        # train=None is just an exception-raising parameter to avoid confused users using the train=False api

        catalog, label_cols = legs_setup(root, split, download, train)

        # paths are not adjusted as cannot be downloaded
        # catalog = _temp_adjust_catalog_paths(catalog)
        # catalog = adjust_catalog_dtypes(catalog, label_cols)

        super().__init__(catalog, label_cols, transform, target_transform)


def legs_setup(root=None, split='train', download=False, train=None):

    if train is not None:
        raise ValueError("This dataset has unlabelled data: use split='train', 'test', 'unlabelled' or 'train+unlabelled' rather than train=False etc")

    assert split in ['train', 'test', 'labelled', 'unlabelled', 'train+unlabelled', 'all']

    # if root is not None:
        # 'Legacy Survey cannot be downloaded - ignoring root {}'.format(root)
        # TODO update for non-manchester users with a manual copy?

    # resources = (
    #     (internal_urls.legs_train_catalog, 'bae2906e337bd114af013d02f3782473'),
    #     (internal_urls.legs_test_catalog, '20919fe512ee8ce4d267790e519fcbf8'),
    #     (internal_urls.legs_unlabelled_catalog, 'fbf287990add34d2249f584325bc9dca'),
    #     # and the images, split into 8gb chunks
    #     (internal_urls.legs_chunk_00, 'd6ca1051b3dd48cfc5c7f0535b403b2d'),
    #     (internal_urls.legs_chunk_01, 'f258ab647cee076ca66288e25f4a778d'),
    #     (internal_urls.legs_chunk_02, '7340e212d5eb38d54ea4d89fff93be81'),
    #     (internal_urls.legs_chunk_03, '33578326983830e2ea5a694757203ae8'),
    #     (internal_urls.legs_chunk_04, '577150bd970ef802a20cd3cce15f656a'),
    #     (internal_urls.legs_chunk_05, '1d88458bf9987c7bf5f21301707c9dd8'),
    #     (internal_urls.legs_chunk_06, '24cf944542e40335f86d7e43468723c0'),
    #     (internal_urls.legs_chunk_07, '583d92b917bd70670d7860e3836cb4a4')
    # )
    resources = (
        (internal_urls.legs_train_catalog, 'bae2906e337bd114af013d02f3782473'),
        (internal_urls.legs_test_catalog, None),
        (internal_urls.legs_unlabelled_catalog, None),
        # and the images, split into 8gb chunks
        (internal_urls.legs_chunk_00, None),
        (internal_urls.legs_chunk_01, None),
        (internal_urls.legs_chunk_02, None),
        (internal_urls.legs_chunk_03, None),
        (internal_urls.legs_chunk_04, None),
        (internal_urls.legs_chunk_05, None),
        (internal_urls.legs_chunk_06, None),
        (internal_urls.legs_chunk_07, '583d92b917bd70670d7860e3836cb4a4')
    )

    if os.path.isdir('/share/nas2'):
        # hardcoded_catalog_root = '/share/nas2/walml/repos/gz-decals-classifiers/data/decals/presplit_catalogs'
        pass
    else:
        hardcoded_catalog_root = '/home/walml/repos/pytorch-galaxy-datasets/roots/legs'  # catalogs only
        root = hardcoded_catalog_root
    downloader = download_utils.DatasetDownloader(root, resources, images_to_spotcheck=['338503/338503_3840.jpg'], image_dirname='')
    if download is True:
        logging.warning('Only downloading catalogs - images are too large to download')
        downloader.download()

    label_cols = label_metadata.decals_all_campaigns_ortho_label_cols
    # useful_columns = label_cols + ['id_str', 'dr8_id', 'brickid', 'objid', 'redshift']

    train_catalog_loc = os.path.join(root, 'legs_all_campaigns_ortho_dr8_only_train_catalog.parquet')
    test_catalog_loc = os.path.join(root, 'legs_all_campaigns_ortho_dr8_only_test_catalog.parquet')
    unlabelled_catalog_loc = os.path.join(root, 'legs_all_campaigns_ortho_dr8_only_unlabelled_catalog.parquet')  # values in label_cols are all 0

    catalogs = []

    if 'all' in split or 'train' in split or ('labelled' in split and 'un' not in split):
        catalogs += [pd.read_parquet(train_catalog_loc)]

    if 'all' in split or 'test' in split or ('labelled' in split and 'un' not in split):
        catalogs += [pd.read_parquet(test_catalog_loc)]

    if 'all' in split or 'unlabelled' in split:
        catalogs += [pd.read_parquet(unlabelled_catalog_loc)]

    logging.info('{} catalogs loaded'.format(split))

    catalog = pd.concat(catalogs, axis=0)
    catalog = catalog.sample(len(catalog), random_state=42).reset_index(drop=True)

    catalog['subfolder'] = catalog['brickid'].astype(str)
    catalog['filename'] = catalog['dr8_id'].astype(str) + '.jpg'
    catalog['file_loc'] = catalog.apply(lambda x: os.path.join(root, downloader.image_dir, x['subfolder'], x['filename']), axis=1)
    logging.info(catalog['file_loc'].iloc[0])


    return catalog, label_cols




if __name__ == '__main__':


    # first download is basically just a convenient way to get the images and canonical catalogs
    legs_dataset = LegsDataset(
        root='whatever',
        train=True,
        download=False  # will fail except on galahad
    )
    legs_catalog = legs_dataset.catalog
    adjusted_catalog = legs_catalog.sample(1000)

    # user will probably tweak and use images/catalog directly for generic galaxy catalog datamodule
    # (which makes its own generic datasets internally)
    datamodule = galaxy_datamodule.GalaxyDataModule(
        label_cols=label_metadata.decals_all_campaigns_label_cols,
        catalog=adjusted_catalog
    )

    datamodule.prepare_data()
    datamodule.setup()
    for images, labels in datamodule.train_dataloader():
        print(images.shape, labels.shape)
        break
        
