# import logging
# import os

# import pandas as pd

# from galaxy_datasets.shared import download_utils
# from galaxy_datasets.check_internal_urls import INTERNAL_URLS_EXIST
# if not INTERNAL_URLS_EXIST:
#     raise FileNotFoundError
# from galaxy_datasets.shared import internal_urls

# # TODO could eventually refactor this out of Zoobot as well
# from zoobot.shared import label_metadata

# """
# Downloads DESI galaxies below z<0.1
# Includes GZ DESI labels where available (identical split to )

# Intended for active learning experiments simulated using GZ DESI
# """

# def desi_low_z(root=None, split='train', download=False, train=None):

#     if train is not None:
#         raise ValueError("This dataset has unlabelled data: use split='train', 'test', 'unlabelled' or 'train+unlabelled' rather than train=False etc")

#     assert split in ['train', 'test', 'labelled', 'unlabelled', 'train+unlabelled', 'all']

#     # if root is not None:
#         # 'Legacy Survey cannot be downloaded - ignoring root {}'.format(root)
#         # TODO update for non-manchester users with a manual copy?


    



#     # resources = (
#     #     (internal_urls.legs_train_catalog, '2364a714a1339f020587d374c1838418'),
#     #     (internal_urls.legs_test_catalog, '647bd46f53a06f13eb4df25311ccb9d3'),
#     #     (internal_urls.legs_unlabelled_catalog, '37c9f07b5c058f55d515d2df08ff132a'),
#     #     # and the images, split into 8gb chunks
#     #     (internal_urls.legs_chunk_00, 'd6ca1051b3dd48cfc5c7f0535b403b2d'),
#     #     (internal_urls.legs_chunk_01, 'f258ab647cee076ca66288e25f4a778d'),
#     #     (internal_urls.legs_chunk_02, '7340e212d5eb38d54ea4d89fff93be81'),
#     #     (internal_urls.legs_chunk_03, '33578326983830e2ea5a694757203ae8'),
#     #     (internal_urls.legs_chunk_04, '577150bd970ef802a20cd3cce15f656a'),
#     #     (internal_urls.legs_chunk_05, '1d88458bf9987c7bf5f21301707c9dd8'),
#     #     (internal_urls.legs_chunk_06, '24cf944542e40335f86d7e43468723c0'),
#     #     (internal_urls.legs_chunk_07, '583d92b917bd70670d7860e3836cb4a4')
#     # )
#     resources = (
#         (internal_urls.legs_train_catalog, '2364a714a1339f020587d374c1838418'),
#         (internal_urls.legs_test_catalog, None),
#         (internal_urls.legs_unlabelled_catalog, None),
#         # and the images, split into 8gb chunks
#         (internal_urls.legs_chunk_00, None),
#         (internal_urls.legs_chunk_01, None),
#         (internal_urls.legs_chunk_02, None),
#         (internal_urls.legs_chunk_03, None),
#         (internal_urls.legs_chunk_04, None),
#         (internal_urls.legs_chunk_05, None),
#         (internal_urls.legs_chunk_06, None),
#         (internal_urls.legs_chunk_07, '583d92b917bd70670d7860e3836cb4a4')
#     )

#     if os.path.isdir('/share/nas2'):
#         # hardcoded_catalog_root = '/share/nas2/walml/repos/gz-decals-classifiers/data/decals/presplit_catalogs'
#         pass
#     else:
#         hardcoded_catalog_root = '/home/walml/repos/pytorch-galaxy-datasets/roots/legs'  # catalogs only
#         root = hardcoded_catalog_root
#     downloader = download_utils.DatasetDownloader(root, resources, images_to_spotcheck=['338503/338503_3840.jpg'], image_dirname='')
#     if download is True:
#         logging.warning('Only downloading catalogs - images are too large to download')
#         downloader.download()

#     label_cols = label_metadata.decals_all_campaigns_ortho_label_cols
#     # useful_columns = label_cols + ['id_str', 'dr8_id', 'brickid', 'objid', 'redshift']

#     train_catalog_loc = os.path.join(root, 'legs_all_campaigns_ortho_dr8_only_train_catalog.parquet')
#     test_catalog_loc = os.path.join(root, 'legs_all_campaigns_ortho_dr8_only_test_catalog.parquet')
#     unlabelled_catalog_loc = os.path.join(root, 'legs_all_campaigns_ortho_dr8_only_unlabelled_catalog.parquet')  # values in label_cols are all 0

#     catalogs = []

#     if 'all' in split or 'train' in split or ('labelled' in split and 'un' not in split):
#         catalogs += [pd.read_parquet(train_catalog_loc)]

#     if 'all' in split or 'test' in split or ('labelled' in split and 'un' not in split):
#         catalogs += [pd.read_parquet(test_catalog_loc)]

#     if 'all' in split or 'unlabelled' in split:
#         catalogs += [pd.read_parquet(unlabelled_catalog_loc)]

#     logging.info('{} catalogs loaded'.format(split))

#     catalog = pd.concat(catalogs, axis=0)
#     catalog = catalog.sample(len(catalog), random_state=42).reset_index(drop=True)

#     catalog['subfolder'] = catalog['brickid'].astype(str)
#     catalog['filename'] = catalog['dr8_id'].astype(str) + '.jpg'
#     catalog['file_loc'] = catalog.apply(lambda x: os.path.join(downloader.image_dir, x['subfolder'], x['filename']), axis=1)
#     logging.info(catalog['file_loc'].iloc[0])


#     return catalog, label_cols
