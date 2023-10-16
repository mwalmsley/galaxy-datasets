import json

import numpy as np
from PIL import Image

from streaming import MDSWriter, StreamingDataset
# pip install mosaicml-streaming

from torch.utils.data import DataLoader

from galaxy_datasets import gz2


def write_shards(write_dir):

    df, label_cols = gz2(
        root='/home/walml/repos/galaxy-datasets/roots/gz2',
        train=True,
        download=True
    )

    columns = {
        'id_str': 'str',
        'img': 'jpeg',
        'label_cols': 'json'
    }
    compression = None  # jpg-encoded anyway
    hashes = ['xxh32']  # non-cryptographic fast hash, for quick validation

    with MDSWriter(
        out=write_dir, columns=columns, compression=compression, hashes=hashes, size_limit='128Mb'
        ) as out:

        for _, galaxy in df.iterrows():
            
            # actually it wants a PIL image specifically
            # img = cv2.imread(galaxy['file_loc'])
            # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = Image.open(galaxy['file_loc'])

            id_str = str(galaxy['id_str'])
            label_col_json = json.dumps(galaxy[label_cols].to_dict())

            sample = {
                'id_str': id_str,
                'img': img,
                'label_cols': label_col_json
            }

            out.write(sample)


class CustomDataset(StreamingDataset):
    def __init__(self, local, remote, **kwargs):
        super().__init__(local=local, remote=remote, **kwargs)

    def __getitem__(self, idx: int):
        sample = super().__getitem__(idx)
        
        sample['img'] = np.array(sample['img']).astype(np.float32)

        return sample


def read_shards(shard_dir, seed):
    
    local = '/tmp/cache'
    remote = shard_dir

    # https://docs.mosaicml.com/projects/streaming/en/latest/fundamentals/shuffling.html
    # all defaults, but I checked they make sense for me
    streaming_dataset = CustomDataset(local, remote, batching_method='random', sampling_method='balanced', shuffle_algo='py1s', shuffle_seed=seed)

    dataloader = DataLoader(dataset=streaming_dataset, batch_size=16, num_workers=6)        

    for batch in dataloader:
         print(batch['id_str'])
         print(batch['label_cols'][0])
         print(batch['img'].shape)
         break


if __name__ == '__main__':
        
        seed = 42

        shard_dir = '/home/walml/repos/galaxy-datasets/roots/sharded/gz2'
        # write_shards(shard_dir)
        read_shards(shard_dir, seed)
