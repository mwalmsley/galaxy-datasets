from typing import Callable
import logging
import torch.utils.data
import numpy as np
import lightning as L
from itertools import islice

import webdataset as wds

from galaxy_datasets import transforms

# https://github.com/webdataset/webdataset-lightning/blob/main/train.py
class WebDataModule(L.LightningDataModule):
    def __init__(
            self,
            train_urls=None,
            val_urls=None,
            test_urls=None,
            predict_urls=None,
            label_cols=None,
            # hardware
            batch_size=64,
            num_workers=4,
            prefetch_factor=4,
            cache_dir=None,
            train_transform: Callable=None,
            inference_transform: Callable=None
            ):
        super().__init__()

        self.train_urls = train_urls
        self.val_urls = val_urls
        self.test_urls = test_urls
        self.predict_urls = predict_urls

        self.set_dataset_size_attributes(train_urls, val_urls, test_urls, predict_urls)

        self.label_cols = label_cols

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.prefetch_factor = prefetch_factor

        self.cache_dir = cache_dir

        self.train_transform = train_transform
        self.inference_transform = inference_transform

        for url_name in ['train', 'val', 'test', 'predict']:
            urls = getattr(self, f'{url_name}_urls')
            if urls is not None:
                logging.info(f"{url_name} (before hardware splits) = {len(urls)} e.g. {urls[0]}", )

        logging.info(f"batch_size: {self.batch_size}, num_workers: {self.num_workers}")

    def set_dataset_size_attributes(self, train_urls=None, val_urls=None, test_urls=None, predict_urls=None):
        if train_urls is not None:
            # assume the size of each shard is encoded in the filename as ..._{size}.tar
            self.train_size = interpret_dataset_size_from_urls(train_urls)
        if val_urls is not None:
            self.val_size = interpret_dataset_size_from_urls(val_urls)
        if test_urls is not None:
            self.test_size = interpret_dataset_size_from_urls(test_urls)
        if predict_urls is not None:
            self.predict_size = interpret_dataset_size_from_urls(predict_urls)

    # def make_image_transform(self, mode="train"):
    #     # only used if you don't explicitly pass a transform to the datamodule
    # NOW DEPRECATED because we don't use albumentations

    #     augmentation_transform = transforms.default_transforms(
    #         crop_scale_bounds=self.crop_scale_bounds,
    #         crop_ratio_bounds=self.crop_ratio_bounds,
    #         resize_after_crop=self.resize_after_crop,
    #         pytorch_greyscale=self.greyscale,
    #         to_float=False,  # True was wrong, webdataset rgb decoder already converts to 0-1 float
    #         # TODO now changed on dev branch will be different for new model training runs
    #         # this compose will now return a Tensor object, default transform was updated
    #         to_tensor=True
    #     )  # A.Compose object

    #     # logging.warning('Minimal augmentations for speed test')
    #     # augmentation_transform = transforms.fast_transforms(
    #     #     resize_after_crop=self.resize_after_crop,
    #     #     pytorch_greyscale=not self.color
    #     # )  # A.Compose object

    #     def do_transform(img):
    #         # img is 0-1 np array, intended for albumentations
    #         assert img.shape[2] < 4  # 1 or 3 channels in shape[2] dim, i.e. numpy/pil HWC convention
    #         # if not, check decode mode is 'rgb' not 'torchrgb'
    #         # default augmentation now returns CHW tensor
    #         return augmentation_transform(image=np.array(img))["image"]
    #     return do_transform


    def make_loader(self, urls, mode="train"):
        logging.info('Making loader with mode {}'.format(mode))

        dataset_size = getattr(self, f'{mode}_size')
        if mode == "train":
            shuffle = min(dataset_size, 5000)
        else:
            assert mode in ['val', 'test', 'predict'], mode
            shuffle = 0

        if self.train_transform is None:
            logging.info('Using default transform')
            raise NotImplementedError('Deprecated')
            # transform_image = self.make_image_transform()
        # else:
            # logging.info('Ignoring other arguments to WebDataModule and using directly-passed transforms')

        transform_image = self.train_transform if mode == 'train' else self.inference_transform

        transform_label = dict_to_label_cols_factory(self.label_cols)

        dataset =  wds.WebDataset(urls, cache_dir=self.cache_dir, shardshuffle=shuffle>0, nodesplitter=nodesplitter_func)
        # https://webdataset.github.io/webdataset/multinode/ 
        # WDS 'knows' which worker it is running on and selects a subset of urls accordingly
           
        if shuffle > 0:
            dataset = dataset.shuffle(shuffle)

        # this controls how webdataset decodes the images, either rgb or torch tensors, see above
        decode_mode = 'torchrgb'  # tensor, for torchvision. Set pil_to_tensor=False in torchvision transforms, already tensor
        # decode_mode = 'rgb' # loads 0-1 np.array, for albumentations
        dataset = dataset.decode(decode_mode)

        # now the webdataset needs to be unpacked (into tensor image, or tensor image, tensor label)
        # and transformed with whatever you passed to transform_image
    
        if mode == 'predict':
            if self.label_cols != ['id_str']:
                logging.info('Will return images only')
                # dataset = dataset.extract_keys('image.jpg').map(transform_image)
                dataset = dataset.to_tuple('image.jpg').map_tuple(transform_image)  # (im,) tuple. But map applied to all elements
                # .map(get_first)
            else:
                logging.info('Will return id_str only')
                dataset = dataset.to_tuple('__key__')
        else:
            
            dataset = (
                dataset.to_tuple('image.jpg', 'labels.json')
                .map_tuple(transform_image, transform_label)
            )

        # torch collate stacks dicts nicely while webdataset only lists them
        # so use the torch collate instead
        dataset = dataset.batched(self.batch_size, torch.utils.data.default_collate, partial=False) 

        # temp hack instead
        if mode in ['train', 'val']:
            assert dataset_size % self.batch_size == 0, (dataset_size, self.batch_size, dataset_size % self.batch_size)
        # for test/predict, always single GPU anyway

        # if mode == "train":
            # ensure same number of batches in all clients
            # loader = loader.ddp_equalize(dataset_size // self.batch_size)
            # print("# loader length", len(loader))

        loader = webdataset_to_webloader(dataset, self.num_workers, self.prefetch_factor)

        return loader

    def train_dataloader(self):
        return self.make_loader(self.train_urls, mode="train")

    def val_dataloader(self):
        return self.make_loader(self.val_urls, mode="val")

    def test_dataloader(self):
        return self.make_loader(self.test_urls, mode="test")
    
    def predict_dataloader(self):
        return self.make_loader(self.predict_urls, mode="predict")

def identity(x):
    return x

def nodesplitter_func(urls):
    urls_to_use = list(wds.split_by_node(urls))  # rely on WDS for the hard work
    rank, world_size, worker, num_workers = wds.utils.pytorch_worker_info()
    logging.debug(
        f'''
        Splitting urls within webdatamodule with WORLD_SIZE: 
        {world_size}, RANK: {rank}, WORKER: {worker} of {num_workers}\n
        URLS: {len(urls_to_use)} (e.g. {urls_to_use[0]})\n\n)
        '''
        )
    return urls_to_use

def interpret_shard_size_from_url(url):
    assert isinstance(url, str), TypeError(url)
    return int(url.rstrip('.tar').split('_')[-1])

def interpret_dataset_size_from_urls(urls):
    return sum([interpret_shard_size_from_url(url) for url in urls])

def get_first(x):
    return x[0]

def custom_collate(x):
    if isinstance(x, list) and len(x) == 1:
        x = x[0]
    return torch.utils.data.default_collate(x)


def webdataset_to_webloader(dataset, num_workers, prefetch_factor):
    loader = wds.WebLoader(
            dataset,
            batch_size=None,  # already batched
            shuffle=False,  # already shuffled
            num_workers=num_workers,
            pin_memory=True,
            prefetch_factor=prefetch_factor
        )

    # loader.length = dataset_size // batch_size
    return loader


def dict_to_label_cols_factory(label_cols=None):
    # converts from dict to vector of counts
    if label_cols is not None:
        def label_transform(label_dict):
            return torch.from_numpy(np.array([label_dict.get(col, 0) for col in label_cols])).double()  # gets cast to int in zoobot loss
        return label_transform
    else:
        return identity  # do nothing


# Used for hybrid pretraining
def dict_to_filled_dict_factory(label_cols):
    logging.info(f'label cols: {label_cols}')
    # might be a little slow, but very safe
    def label_transform(label_dict: dict):

        # modifies inplace with 0 iff key missing
        # [label_dict.setdefault(col, 0) for col in label_cols]

        for col in label_cols:
            label_dict[col] = label_dict.get(col, 0)

        # label_dict_with_default = defaultdict(0)
        # label_dict_with_default.update(label_dict)

        return label_dict
    return label_transform


# just for debugging
def load_wds_with_webdatamodule(save_loc, label_cols, batch_size=16, max_to_load=3):
    wdm = WebDataModule(
        train_urls=save_loc,
        val_urls=save_loc,  # not used
        # train_size=len(train_catalog),
        # val_size=0,
        label_cols=label_cols,
        num_workers=1,
        batch_size=batch_size
    )
    wdm.setup('fit')

    if max_to_load is not None:
        sample_iterator =islice(wdm.train_dataloader(), 0, max_to_load)
    else:
        sample_iterator = wdm.train_dataloader()
    for sample in sample_iterator:
        images, labels = sample
        logging.info(images.shape)
        # logging.info(len(labels))  # list of dicts
        logging.info(labels.shape)

