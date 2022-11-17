import os
from functools import partial
from typing import List
import logging

import pandas as pd
import tensorflow as tf


def get_image_dataset(
    image_paths: List,
    labels=None,
    check_valid_paths=True,
    # new preprocessing options now here
    requested_img_size=None,
    greyscale=False,
    permute_channels=False,
    include_id_str=False) -> tf.data.Dataset:
    """
    Load images in a folder as a tf.data dataset
    Supports jpeg (note the e) and png

    Args:
        image_paths (list): list of image paths to load
        requested_img_size (int or None): e.g. 256 for 256x256x3 image. Assumed square. Will resize if size on disk != this. No resizing if None.
        labels (list or None): If not None, include labels in dataset (see Returns). Must be equal length to image_paths. Defaults to None.
        greyscale (bool): 
        permute_channels (bool):

    Raises:
        FileNotFoundError: at least one path does not match an existing file

    Returns:
        tf.data.Dataset: yielding batches with 'image' key for image array, 'id_str' for the image path, and 'label' if ``labels`` was provided.
    """
    
    assert len(image_paths) > 0
    if isinstance(image_paths, pd.Series):
        image_paths = list(image_paths)
    example_path = image_paths[0]
    assert isinstance(example_path, str)
    logging.info('Image paths to load as dataset: {}'.format(len(image_paths)))

    if check_valid_paths:
        logging.info('Checking if all paths are valid')
        missing_paths = [path for path in image_paths if not os.path.isfile(path)]
        if missing_paths:
            raise FileNotFoundError(f'Missing {len(missing_paths)} images e.g. {missing_paths[0]}')
        logging.info('All paths exist')
    else:
        logging.warning('Skipping valid path check')

    file_format = example_path.split('.')[-1]  # assume all same format

    # load a single image to check the shape
    test_image = load_image_as_element(example_path)['image']
    size_on_disk = test_image.numpy().shape[0]  # x dimension (XYC convention, not yet batched)
    if requested_img_size is None:
        logging.info('No specific image size requested on load - not resizing (until augmentations')
        resize_size = None
    elif size_on_disk == requested_img_size:
        logging.info('Image size on disk matches requested_img_size of {}, skipping resizing'.format(requested_img_size))  # x dimension of first image, first y index, first channel
        resize_size = None
    else:
        logging.warning('Resizing images from disk size {} to requested size {}'.format(size_on_disk, requested_img_size))
        resize_size = requested_img_size

    path_ds = tf.data.Dataset.from_tensor_slices([str(path) for path in image_paths])

    # will yield elements of {'image': image, 'id_str': path}
    image_ds = path_ds.map(lambda x: load_image_as_element(x, mode=file_format), num_parallel_calls=tf.data.AUTOTUNE)  # keep determinstic though

    # applies preprocessing transforms to x['image']: greyscaling/permuting, resizing
    # TODO could join these into the above function for speed, if needed
    image_ds = image_ds.map(lambda x: preprocess_image_in_element(x, resize_size=resize_size, greyscale=greyscale, permute_channels=permute_channels), num_parallel_calls=tf.data.AUTOTUNE)

    if labels is not None:
        assert len(labels) == len(image_paths)

        # if isinstance(labels[0], dict):
        #     # assume list of dicts, each representing one datapoint e.g. [{'feat_a': 1, 'feat_b': 2}, {'feat_a': 12, 'feat_b': 42}]
        #     # reshape to columnlike dict e.g. {'feat_a': [1, 12], 'feat_b: [2, 42]} because that's what tf supports
        #     # (could pass this directly, but inputs tend to be easier to handle row-wise for e.g. shuffling etc)
        #     label_ds = tf.data.Dataset.from_tensor_slices(pd.DataFrame.from_dict(labels).to_dict(orient="list"))
        # else:
            # make it a dict anyway for consistency, keyed by 'label' instead of e.g. 'feat_a', 'feat_b'
        label_ds = tf.data.Dataset.from_tensor_slices({'label': labels})

        # label_dict is {'label': (256)} or {'feat_a': (256), 'feat_b': (256)}
        # image_dict is {'id_str': some_id 'image': (image)}
        # merge the two dicts to create {'id_str': ..., 'image': ..., 'feat_a': ..., 'feat_b': ...}
        image_ds = tf.data.Dataset.zip((image_ds, label_ds)).map(lambda image_dict, label_dict: {**image_dict, **label_dict})
        # now yields {'image': , 'id_str': , 'label': } batched dicts

    # now choose how to unpack
    # (yes, I could probably do this faster/in fewer steps, but this is very readable)
    if include_id_str:
        if labels is not None:
            image_ds = image_ds.map(lambda x: (x['image'], x['label'], x['id_str']))
        else:
            image_ds = image_ds.map(lambda x: (x['image'], x['id_str']))
    else:
        assert labels is not None
        image_ds = image_ds.map(lambda x: (x['image'], x['label']))

    return image_ds


# https://stackoverflow.com/questions/62544528/tensorflow-decodejpeg-expected-image-jpeg-png-or-gif-got-unknown-format-st?rq=1
def load_image_as_element(loc, mode='png'):
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


def preprocess_image_in_element(element, resize_size=None, greyscale=False, permute_channels=False):
    # now applies all preprocessing

    # element['image'] must be 0-1 floats, not 0-255 ints, or clipping will ruin
    image, id_str = element['image'], element['id_str']  # unpack from dict

    if greyscale:
        # new channel dimension of 1
        image = tf.reduce_mean(input_tensor=image, axis=-1, keepdims=True)
        # assert images.shape[1] == input_size
        # assert images.shape[2] == input_size
        # assert images.shape[3] == 1
        # tf.summary.image('b_make_greyscale', images)
    else:
        if permute_channels:
            image = tf.map_fn(permute_channels, image)  # map to each image in element
        else:
            image = tf.identity(image)
            
    if resize_size is not None:
        image = resize_image_with_tf(image , size=resize_size)   # initial size = after resize from image on disk (e.g. 424 for GZ pngs) but before crop/zoom
        image = tf.clip_by_value(image, 0., 1.)  # resizing can cause slight change in min/max

    return {'image': image, 'id_str': id_str}  # pack back into dict


def resize_image_with_tf(image, size):
    # May cause values outside 0-255 margins
    # allows for image dimension, or no image dimension
    # May be slow. Ideally, resize the images beforehand on disk (or save as TFRecord, see make_shards.py and tfrecord_datasets.py)
    return tf.image.resize(image, (size, size), method=tf.image.ResizeMethod.LANCZOS3, antialias=True)


def permute_channels(im):
    assert tf.shape(im)[-1] == 3
    # tf.random.shuffle shuffles 0th dimension, so need to temporarily swap channel to 0th, shuffle, and swap back
    return tf.transpose(tf.random.shuffle(tf.transpose(im, perm=[2, 1, 0])), perm=[2, 1, 0])


def add_transforms_to_dataset(dataset, transforms):
    # only works if dataset yields either (images, labels) or (images, id_str), not (images, label_str, id_str)
    # TODO make work with extra element - a little fiddly due to unpacking in graph mode, though

    # use closure to pass in transforms (which cannot be passed via inp as not tensor-like)
    def process_data(images, labels):
        aug_imgs = tf.numpy_function(func=aug_fn, inp=[images], Tout=tf.float32)
        return aug_imgs, labels
        # return (aug_imgs) + args  # equivalent to aug_imgs, *args, but works prior to Python 3.8

    def aug_fn(image):
        # albumentations.transforms expects input like image=x
        # and returns {'image': image} dict
        return transforms(image=image)["image"]

    # TODO specify output_size

    return dataset.map(
        partial(process_data),
        num_parallel_calls=tf.data.experimental.AUTOTUNE
    )

