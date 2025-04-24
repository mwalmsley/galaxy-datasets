import pytest

import numpy as np
import torch
from galaxy_datasets import transforms


# def custom_jwst_config():
#     transform_config = transforms.default_view_config()
#     transform.fixed_crop = 

@pytest.fixture
def image_batch_tensor():
    return (torch.rand(16, 3, 300, 300) * 255.).to(torch.uint8)  # 16 images, 3 channels, 224x224 pixels

@pytest.fixture
def image_pil():
    from PIL import Image
    return Image.fromarray(np.uint8(np.random.rand(300, 300, 3)*255))

@pytest.fixture(params=[transforms.default_view_config, transforms.minimal_view_config, transforms.fast_view_config])
def base_transform_config(request):
    return request.param()

@pytest.fixture(params=[True, False])
def grayscale(request):
    return request.param

@pytest.fixture(params=[True, False])
def pil_to_tensor(request):
    return request.param

@pytest.fixture
def transform_config(base_transform_config, grayscale, pil_to_tensor):
    # Create a copy of the base config
    config = base_transform_config.copy()
    config.greyscale = grayscale
    config.pil_to_tensor = pil_to_tensor
    return config


def test_GalaxyViewTransform_tensor(image_batch_tensor, transform_config):
    transform_config.pil_to_tensor = False  # starting with a tensor
    transform = transforms.GalaxyViewTransform(transform_config)
    transformed_batch = transform(image_batch_tensor)
    standard_image_checks(transformed_batch, transform_config)

def test_GalaxyViewTransform_pil(image_pil, transform_config):
    transform_config.pil_to_tensor = True  # starting with a list of PIL images
    transform = transforms.GalaxyViewTransform(transform_config)
    transformed_image = transform(image_pil)
    standard_image_checks(transformed_image, transform_config)


def standard_image_checks(transformed_image_or_batch, transform_config):
    # either image or batch, in tensor format
    if transformed_image_or_batch.ndim == 4:
        # batch of images, NCHW format
        assert transformed_image_or_batch.shape[0] == 16, f"Expected batch size 16, but got {transformed_image_or_batch.shape[0]}"
        # and take the first image
        transformed_image = transformed_image_or_batch[0]
    else:
        # single image, CHW format
        transformed_image = transformed_image_or_batch

    if transform_config.greyscale:
        # check channel dimension is 1
        assert transformed_image.shape[0] == 1, f"Expected 1 channel, but got {transformed_image.shape[0]}"
    else:
        # check channel dimension is 3
        assert transformed_image.shape[0] == 3, f"Expected 3 channels, but got {transformed_image.shape[0]}"

    assert transformed_image.shape[1] == transform_config.output_size, f"Expected height {transform_config.output_size}, but got {transformed_image.shape[1]}"
    assert transformed_image.shape[2] == transform_config.output_size, f"Expected width {transform_config.output_size}, but got {transformed_image.shape[2]}"
    

    standard_image_normalization_checks(transformed_image)


def standard_image_normalization_checks(transformed_image):
    # either image or batch, in tensor format
    # Check the batch size didn't change
    assert transformed_image.max() <= 1.0, "Transformed image max value should be <= 1.0"
    assert transformed_image.min() >= 0.0, "Transformed image min value should be >= 0.0"
    assert transformed_image.max() >= 1/256, "Transformed image max value should be >= 1/256, possible scaling issue"
    assert transformed_image.dtype == torch.float32, "Transformed image should be of type float32"



# grab demo rings

# if __name__ == '__main__':

#     from galaxy_datasets import demo_rings

#     demo_rings(root='tests/data', train=True, download=True)

