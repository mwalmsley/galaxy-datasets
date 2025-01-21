import logging
from typing import Union, Callable
import os
import cv2
import json
from itertools import islice


import albumentations as A

import tqdm
import numpy as np
import pandas as pd
from PIL import Image  # necessary to avoid PIL.Image error assumption in web_datasets

from galaxy_datasets.transforms import default_transforms

import webdataset as wds


def catalogs_to_webdataset(dataset_name, wds_dir, label_cols, train_catalog, test_catalog, sparse_label_df=None, divisor=512, overwrite=False):
    for (catalog_name, catalog) in [('train', train_catalog), ('test', test_catalog)]:
        n_shards = len(catalog) // divisor
        logging.info(n_shards)

        catalog = catalog[:n_shards*divisor]
        logging.info(len(catalog))

        save_loc = f"{wds_dir}/{dataset_name}/{dataset_name}_{catalog_name}.tar"  # .tar replace automatically
        
        df_to_wds(catalog, label_cols, save_loc, n_shards=n_shards, sparse_label_df=sparse_label_df, overwrite=overwrite)

    


def df_to_wds(df: pd.DataFrame, label_cols, save_loc: str, n_shards: int, sparse_label_df=None, overwrite=False):

    assert '.tar' in save_loc
    df['id_str'] = df['id_str'].astype(str).str.replace('.', '_')
    if sparse_label_df is not None:
        logging.info(f'Using sparse label df: {len(sparse_label_df)}')
    shard_dfs = np.array_split(df, n_shards)
    logging.info(f'shards: {len(shard_dfs)}. Shard size: {len(shard_dfs[0])}')

    transforms_to_apply = [
        # below, for 224px fast training fast augs setup
        # A.Resize(
        #     height=350,  # now more aggressive, 65% crop effectively
        #     width=350,  # now more aggressive, 65% crop effectively
        #     interpolation=cv2.INTER_AREA  # slow and good interpolation
        # ),
        # A.CenterCrop(
        #     height=224,
        #     width=224,
        #     always_apply=True
        # ),
        # below, for standard training default augs
        # small boundary trim and then resize expecting further 224px crop
        # we want 0.7-0.8 effective crop
        # in augs that could be 0.x-1.0, and here a pre-crop to 0.8 i.e. 340px
        # but this would change the centering
        # let's stick to small boundary crop and 0.75-0.85 in augs

        # turn these off for current euclidized images, already 300x300
        # A.CenterCrop(
        #     height=400,
        #     width=400,
        #     always_apply=True
        # ),
        # A.Resize(
        #     height=300,
        #     width=300,
        #     interpolation=cv2.INTER_AREA  # slow and good interpolation
        # )

        # with GZ Euclid, apply no transforms, the images are saved at native size already
    ]
    transform = A.Compose(transforms_to_apply)
    # transform = None

    for shard_n, shard_df in tqdm.tqdm(enumerate(shard_dfs), total=len(shard_dfs)):
        shard_save_loc = save_loc.replace('.tar', f'_{shard_n}_{len(shard_df)}.tar')
        if overwrite or not(os.path.isfile(shard_save_loc)):
            if sparse_label_df is not None:
                shard_df = pd.merge(shard_df, sparse_label_df, how='left', validate='one_to_one', suffixes=('', '_badlabelmerge'))  # type: ignore # auto-merge

            assert not any(shard_df[label_cols].isna().max()) # type: ignore

            # logging.info(shard_save_loc)
            sink = wds.TarWriter(shard_save_loc)
            for _, galaxy in shard_df.iterrows(): # type: ignore
                try:
                    sink.write(galaxy_to_wds(galaxy, label_cols, transform=transform))
                except Exception as e:
                    logging.critical(galaxy)
                    raise(e)
            sink.close()


def galaxy_to_wds(galaxy: pd.Series, label_cols: Union[list[str],None]=None, metadata_cols: Union[list, None]=None, transform: Union[Callable, None]=None):

    assert os.path.isfile(galaxy['file_loc']), galaxy['file_loc']
    im = cv2.imread(galaxy['file_loc'])
    # cv2 loads BGR for 'history', fix
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB) 
    assert not np.any(np.isnan(np.array(im))), galaxy['file_loc']

    id_str = str(galaxy['id_str'])

    if transform is not None:
        im = transform(image=im)['image']

    dict_to_save = {
        "__key__": id_str,  # silly wds bug where if __key__ ends .jpg, all keys get jpg. prepended?! use id_str instead
        "image.jpg": im,
    }
    if label_cols is not None:
        dict_to_save['labels.json'] = json.dumps(galaxy[label_cols].to_dict())
    if metadata_cols is not None:
        dict_to_save['metadata.json'] = json.dumps(galaxy[metadata_cols].to_dict())
    return dict_to_save
    
    # huggingface doesn't like empty keys

    # if label_cols is None:
    #     labels = json.dumps({})
    # else:
    #     labels = json.dumps(galaxy[label_cols].to_dict())
  
    # if metadata_cols is None:
    #     metadata = json.dumps({})
    # else:
    #     
    
    # return {
    #     "image.jpg": im,
    #     "labels.json": labels,
    #     "metadata.json": metadata
    # }


# just for debugging
def load_wds_directly(wds_loc, max_to_load=3):

    dataset = wds.WebDataset(wds_loc) \
    .decode("rgb")

    if max_to_load is not None:
        sample_iterator = islice(dataset, 0, max_to_load)
    else:
        sample_iterator = dataset
    for sample in sample_iterator:
        logging.info(sample['__key__'])     
        logging.info(sample['image.jpg'].shape)  # .decode(jpg) converts to decoded to 0-1 RGB float, was 0-255
        logging.info(type(sample['labels.json']))  # automatically decoded


# just for debugging
def load_wds_with_augmentation(wds_loc):

    augmentation_transform = default_transforms()  # A.Compose object
    def do_transform(img):
        return np.transpose(augmentation_transform(image=np.array(img))["image"], axes=[2, 0, 1]).astype(np.float32)

    dataset = wds.WebDataset(wds_loc) \
        .decode("rgb") \
        .to_tuple('image.jpg', 'labels.json') \
        .map_tuple(do_transform, identity)
    
    for sample in islice(dataset, 0, 3):
        logging.info(sample[0].shape)
        logging.info(sample[1])


def identity(x):
    # no lambda to be pickleable
    return x



def make_mock_wds(save_dir: str, label_cols: list, n_shards: int, shard_size: int):
    counter = 0
    shards = [os.path.join(save_dir, f'mock_shard_{shard_n}_{shard_size}.tar') for shard_n in range(n_shards)]
    for shard in shards:
        sink = wds.TarWriter(shard)
        for galaxy_n in range(shard_size):
            data = {
                "__key__": f'id_{galaxy_n}',
                "image.jpg": (np.random.rand(424, 424)*255.).astype(np.uint8),
                "labels.json": json.dumps(dict(zip(label_cols, [np.random.randint(low=0, high=10) for _ in range(len(label_cols))])))
            }
            sink.write(data)
            counter += 1
    print(counter)
    return shards


if __name__ == '__main__':

    save_dir = '/home/walml/repos/temp'
    from galaxy_datasets.shared import label_metadata
    label_cols = label_metadata.decals_all_campaigns_ortho_label_cols

    make_mock_wds(save_dir, label_cols, n_shards=4, shard_size=512)
