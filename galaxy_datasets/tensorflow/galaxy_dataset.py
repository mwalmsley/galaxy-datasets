# import copy
from typing import List
import logging

import pandas as pd
import tensorflow as tf
# from skimage.transform import warp, AffineTransform, SimilarityTransform

import os
import logging

import pandas as pd
import tensorflow as tf

import matplotlib.pyplot as plt


def get_image_dataset(
    image_paths: List,
    file_format: str,
    requested_img_size: int,
    batch_size: int,
    labels=None,
    check_valid_paths=True,
    shuffle=False,
    drop_remainder=False,
    # new preprocessing options now here
    greyscale=False,
    permute_channels=False) -> tf.data.Dataset:
    """
    Load images in a folder as a tf.data dataset
    Supports jpeg (note the e) and png

    Args:
        image_paths (list): list of image paths to load
        file_format (str): image format e.g. png, jpeg
        requested_img_size (int): e.g. 256 for 256x256x3 image. Assumed square. Will resize if size on disk != this.
        batch_size (int): batch size to use when grouping images into batches
        labels (list or None): If not None, include labels in dataset (see Returns). Must be equal length to image_paths. Defaults to None.

    Raises:
        FileNotFoundError: at least one path does not match an existing file

    Returns:
        tf.data.Dataset: yielding batches with 'image' key for image array, 'id_str' for the image path, and 'label' if ``labels`` was provided.
    """
    
    assert len(image_paths) > 0
    assert isinstance(image_paths[0], str)
    logging.info('Image paths to load as dataset: {}'.format(len(image_paths)))

    if check_valid_paths:
        logging.info('Checking if all paths are valid')
        missing_paths = [path for path in image_paths if not os.path.isfile(path)]
        if missing_paths:
            raise FileNotFoundError(f'Missing {len(missing_paths)} images e.g. {missing_paths[0]}')
        logging.info('All paths exist')
    else:
        logging.warning('Skipping valid path check')

    path_ds = tf.data.Dataset.from_tensor_slices([str(path) for path in image_paths])

    image_ds = path_ds.map(lambda x: load_image_file(x, mode=file_format), num_parallel_calls=tf.data.AUTOTUNE)  # keep determinstic though

    # image_ds = image_ds.batch(batch_size, drop_remainder=drop_remainder)

    # check if the image shape matches requested_img_size, and resize if not
    test_image = [image for image in image_ds.take(1)]['image']
    size_on_disk = test_image.numpy().shape[0]  # x dimension (XYC convention, not yet batched)
    if size_on_disk == requested_img_size:
        logging.info('Image size on disk matches requested_img_size of {}, skipping resizing'.format(requested_img_size))  # x dimension of first image, first y index, first channel
    else:
        logging.warning('Resizing images from disk size {} to requested size {}'.format(size_on_disk, requested_img_size))
        image_ds = image_ds.map(lambda x: prepare_image_batch(x, resize_size=requested_img_size, greyscale=greyscale, permute_channels=permute_channels))

    # now returns floats from 0 to 1
    # image_batch = list(image_ds.take(1))[0]
    # print(image_batch)
    # image = image_batch['image'].numpy()[0]
    # print(image.min(), image.max())
    # import matplotlib.pyplot as plt
    # plt.imshow(image)
    # plt.show()
    # exit()

    if labels is not None:
        assert len(labels) == len(image_paths)

        if isinstance(labels[0], dict):
            # assume list of dicts, each representing one datapoint e.g. [{'feat_a': 1, 'feat_b': 2}, {'feat_a': 12, 'feat_b': 42}]
            # reshape to columnlike dict e.g. {'feat_a': [1, 12], 'feat_b: [2, 42]} because that's what tf supports
            # (could pass this directly, but inputs tend to be easier to handle row-wise for e.g. shuffling etc)
            label_ds = tf.data.Dataset.from_tensor_slices(pd.DataFrame.from_dict(labels).to_dict(orient="list"))
        else:
            # make it a dict anyway for consistency, keyed by 'label' instead of e.g. 'feat_a', 'feat_b'
            label_ds = tf.data.Dataset.from_tensor_slices({'label': labels})

        # drop_remainder applied to labels as well, if relevant. Equal length = same drop.
        # label_ds = label_ds.batch(batch_size, drop_remainder=drop_remainder)

        # print(list(label_ds.take(1)))  

        # label_dict is {'label': (256)} or {'feat_a': (256), 'feat_b': (256)}
        # image_dict is {'id_str': some_id 'image': (image)}
        # merge the two dicts to create {'id_str': ..., 'image': ..., 'feat_a': ..., 'feat_b': ...}
        image_ds = tf.data.Dataset.zip((image_ds, label_ds)).map(lambda image_dict, label_dict: {**image_dict, **label_dict})
        # now yields {'image': , 'id_str': , 'label': } batched dicts

    # shuffle must only happen *after* zipping in the labels
    # if shuffle:
    #     image_ds = image_ds.shuffle(buffer_size=5)  # already batched, so buffer is *batches*
    #     # TODO could use interleave etc

    # image_ds = image_ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    return image_ds




# https://stackoverflow.com/questions/62544528/tensorflow-decodejpeg-expected-image-jpeg-png-or-gif-got-unknown-format-st?rq=1
def load_image_file(loc, mode='png'):
    """
    Load an image file from disk to memory.
    *Recently changed to return 0-1 floats not 0-255 floats*

    Args:
        loc (str): Path to image on disk. Includes format e.g. .png.
        mode (str, optional): Image format. Defaults to 'png'.

    Raises:
        ValueError: mode is neither png nor jpeg.

    Returns:
        dict: like {'image': float32 np.ndarray from 0. to 1., 'id_str': ``loc``}
    """
    # values will be 0-255, does not normalise. Happens in preprocessing instead.
    # specify mode explicitly to avoid graph tracing issues
    image = tf.io.read_file(loc)
    if mode == 'png':
        image = tf.image.decode_png(image)
    # rename jpg to jpeg or validation checks in decode_jpg will fail
    # might fail, let's see
    elif (mode == 'jpg') or (mode == 'jpeg'):  
        # TODO also allow jpg
        image = tf.image.decode_jpeg(image)
    else:
        raise ValueError(f'Image filetype mode {mode} not recognised')

    converted_image = tf.cast(image, tf.float32) / 255.  # floats in 0-1 range

    return {'image': converted_image, 'id_str': loc}  # using the file paths as identifiers


def resize_image_batch_with_tf(batch, size):
    # May cause values outside 0-255 margins
    # May be slow. Ideally, resize the images beforehand on disk (or save as TFRecord, see make_shards.py and tfrecord_datasets.py)
    return tf.image.resize(batch, (size, size), method=tf.image.ResizeMethod.LANCZOS3, antialias=True)


def prepare_image_batch(batch, resize_size=None, greyscale=False, permute_channels=False):
    # now applies all preprocessing

    # batch['image'] must be 0-1 floats, not 0-255 ints, or clipping will ruin
    images, id_strs = batch['image'], batch['id_str']  # unpack from dict
    if greyscale:
        # new channel dimension of 1
        images = tf.reduce_mean(input_tensor=images, axis=-1, keepdims=True)
        # assert images.shape[1] == input_size
        # assert images.shape[2] == input_size
        # assert images.shape[3] == 1
        # tf.summary.image('b_make_greyscale', images)
    else:
        if permute_channels:
            images = tf.map_fn(permute_channels, images)  # map to each image in batch
        else:
            images = tf.identity(images)
            
    if resize_size:
        images = resize_image_batch_with_tf(images , size=resize_size)   # initial size = after resize from image on disk (e.g. 424 for GZ pngs) but before crop/zoom
        images = tf.clip_by_value(images, 0., 1.)  # resizing can cause slight change in min/max
    return {'image': images, 'id_str': id_strs}  # pack back into dict


def permute_channels(im):
    assert tf.shape(im)[-1] == 3
    # tf.random.shuffle shuffles 0th dimension, so need to temporarily swap channel to 0th, shuffle, and swap back
    return tf.transpose(tf.random.shuffle(tf.transpose(im, perm=[2, 1, 0])), perm=[2, 1, 0])

from functools import partial
from albumentations import (
    Compose, RandomBrightness, JpegCompression, HueSaturationValue, RandomContrast, HorizontalFlip,
    Rotate
)
AUTOTUNE = tf.data.experimental.AUTOTUNE


# Wrapping this causes weird op error - leave it be. Issue raised w/ tf.
# @tf.function
def unpack_dataset(dataset, label_cols):
    """
    Thin wrapper applying ``unpack_batch`` across dataset. See ``unpack_batch`` for more.

    Args:
        config (PreprocessingConfig): Configuration object defining how 'get_input' should function

    Returns:
        batches of (images, labels) or (images, id_strs)
    """
    return dataset.map(lambda x: unpack_element(x, label_cols), num_parallel_calls=tf.data.AUTOTUNE)


def unpack_element(element, label_cols: List):
    """
    Split element into:
    - tuples of (images, labels) if ``config.label_cols`` is not None
    - tuples of (images, id_strings) otherwise.
    
    Args:
        element (dict): not quite a dict but a tf.data.Dataset element, which can be keyed with element['image'], element['id_str'], and perhaps batch[col] for each col in ``config.label_cols``
        label_cols (list):
    
    Returns:
        tuple: see above
    """
    image = element['image']

    if len(label_cols) == 0:
        logging.warning('No labels requested, returning id_str as labels')
        return image, batch['id_str']
    else:
        # how to unpack labels depends if the user passed a dict of labels, or a list of labels
        # if a list, will be keyed under 'label'
        if len(label_cols) == 1:
            label = element['label']
        # if a dict, will be keyed however the user passed in (assumed to match label_cols TODO)
        else:
            label = tf.stack([element[col] for col in label_cols], axis=1)   # element[cols] appears not to work

    return image, label

# def get_images_from_batch(batch):
#     """
#     Extract images from batch
#     Useful to then manipulate those images.

#     Args:
#         batch (dict): tf.data.Dataset batch with images under 'image' key

#     Returns:
#         tf.Tensor: images of shape ``(batch_size, size, size, channels)``
#     """
#     return tf.cast(batch['image'], tf.float32)  # may automatically read uint8 into float32, but let's be sure


# def get_labels_from_batch(batch, label_cols: List):
#     """
#     Extract labels from batch.

#     Batch will have labels keyed under batch[col] for col in ``label_cols``.
#     Stack those labels into a tf.Tensor that can then be used for e.g. evaluating a model.
#     Order of labels in the tf.Tensor will match that of ``label_cols``.

#     Args:
#         batch (dict): tf.data.Dataset batch
#         label_cols (List): strings for each answer e.g. ['smooth-or-featured_smooth', 'smooth-or-featured_featured-or-disk', etc]

#     Returns:
#         tf.Tensor: labels extracted from batch, of shape (batch_size, num. answers)
#     """


def add_augmentations_to_dataset(dataset):
    # dataset yields (images, _)  where _ might be labels or id_str (passed along unchanged either way)

    transforms = Compose([
            Rotate(limit=40),
            # RandomBrightness(limit=0.1),
            # JpegCompression(quality_lower=85, quality_upper=100, p=0.5),
            # HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.5),
            # RandomContrast(limit=0.2, p=0.5),
            HorizontalFlip(),
        ])

    # use closure to pass in transforms (which cannot be passed via inp as not tensor-like)
    def process_data(images, labels):
        aug_imgs = tf.numpy_function(func=aug_fn, inp=[images], Tout=tf.float32)
        return aug_imgs, labels

    def aug_fn(image_batch):
        aug_data = transforms(image=image_batch)  # transforms expects input like image=image_batch
        aug_img = aug_data["image"]
        # aug_img = tf.cast(aug_img/255.0, tf.float32)
        # aug_img = tf.image.resize(aug_img, size=[img_size, img_size])
        return aug_img

    return dataset.map(partial(process_data), num_parallel_calls=AUTOTUNE).prefetch(AUTOTUNE)


if __name__ == '__main__':

    from galaxy_datasets.prepared_datasets import tidal

    catalog, label_cols = tidal.setup(
        root='/nvme1/scratch/walml/repos/pytorch-galaxy-datasets/roots/tidal',
        train=True,
        download=False)

    dataset = get_image_dataset(
        image_paths=catalog['file_loc'],
        file_format='jpg',
        requested_img_size=64,
        batch_size=32,
        labels=catalog['coarse_tidal_label'],
        greyscale=True,
        permute_channels=False
    )
    # add any further one-time preprocessing here (specifically, prepare_image_batch)

    for batch in dataset.take(1):
        print(batch.keys())
        print(batch['image'].shape)
        print(batch['label'])

    # TODO not totally sure where I want this step to happen
    dataset = unpack_dataset(
        dataset,
        label_cols
    )

    for images, labels in dataset.take(1):
        print(images.shape)
        print(labels)

    transforms = Compose([
        Rotate(limit=40),
        HorizontalFlip(),
    ])

    # example_image = images.numpy()

    # print(example_image.shape)

    # aug_images = transforms(image=example_image)['image']

    # TODO nope, albumentations does NOT expect batches
    # it works on single images (float32 0-1 is okay and preserved)
    # aug_images = transforms(image=example_image)['image']

    # print(aug_images)
    
    # plt.imshow(aug_images)
    # plt.show()

    dataset = add_augmentations_to_dataset(
        dataset
    )

    for image, label in dataset.take(1):
        print(image.shape)
        print(label)

    # and then apply batch and then prefetch, as needed
    dataset = dataset.shuffle(5000).batch(16).prefetch(buffer_size=AUTOTUNE)

    for images, labels in dataset.take(1):
        print(images.shape)
        print(labels)
