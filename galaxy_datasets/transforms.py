import logging

from omegaconf import DictConfig

import torch
import torchvision.transforms.v2 as T
from torchvision.transforms import InterpolationMode

from galaxy_datasets.transforms_custom_torchvision import FixedCrop

class GalaxyViewTransform():
    # most powerful and most general transform options

    def __init__(
        self,
        cfg: DictConfig
    ):
        """
        Transforms expect to act on channel-first tensor. 
        Probably uint8 0-255 for speed, but not necessary.

        Transforms return a channel-first float32 tensor in the range (0, 1)

        Args:
            cfg (DictConfig): options for common torchvision transforms
        """
        logging.info(f'Using view transform config: {cfg}')
        
        interpolation = InterpolationMode[cfg.interpolation_method.upper()]
        antialias = True
        
        transform = []
        
        if cfg.greyscale:
            transform.append(T.Grayscale(num_output_channels=1))

        # transform += [T.ToImage()]  # explicit but not needed, just tells torchvision it's an image (assumed anway)
        # disabled for beluga
        # 'torchvision.transforms.v2' has no attribute 'ToImage'
        # requires 0.16.0, beluga has 0.15.2

        if cfg.fixed_crop:
            # fixed crop with specified corners, useful to pick a single image from Zooniverse grid subjects
            transform.append(FixedCrop(
                **cfg.fixed_crop  # lower_left_x, lower_left_y, upper_right_x, upper_right_y
            ))
            logging.info(f'Using FixedCrop, {cfg.fixed_crop}')

        # shear/perspective transforms happen before any cropping
        if cfg.random_affine:  # will use this for rotation and off-center zoom/crop
            transform.append(T.RandomAffine(
                **cfg.random_affine, interpolation=interpolation
            ))
            transform.append(T.Resize(cfg.output_size, interpolation=interpolation, antialias=antialias))
            logging.info(f'Using RandomAffine with safety crop, {cfg.random_affine}')
            transform.append(T.CenterCrop(cfg.output_size))  # T.Resize is a touch imprecise e.g. (224, 227) after resizing. Do a final center crop for safety because torch requires exact shapes.

        else:  # maybe do perspective/crop, maybe do rotation, maybe both
            logging.info('Not using RandomAffine')

            if cfg.random_perspective:
                logging.info(f'Using RandomPerspective, {cfg.random_perspective}')
                transform.append(T.RandomPerspective(**cfg.random_perspective, interpolation=interpolation))
                transform.append(T.CenterCrop(size=cfg.output_size))
            else: # no affine transform, no perspective shift, so safe to apply simple center or random crops
                if cfg.center_crop:
                    transform.append(T.CenterCrop(
                            size=cfg.output_size
                        ))
                if cfg.random_resized_crop:
                    transform.append(T.RandomResizedCrop(
                                size=cfg.output_size,
                                interpolation=interpolation,
                                antialias=antialias,
                                **cfg.random_resized_crop  # scale only
                            ))
                    # transform.append(T.CenterCrop(cfg.output_size))  # T.RandomResizedCrop is also a touch imprecise e.g. (224, 227) after resizing.

            if cfg.rotation_prob > 0:
                    logging.info(f'Using RandomRotation, {cfg.rotation_prob}')
                    # I ASSUME this doesn't change the shape?
                    transform.append(T.RandomRotation(degrees=90, interpolation=interpolation, p=cfg.rotation_prob))


        # flip and rotate. safe after any geometric transforms.
        transform.append(T.RandomHorizontalFlip(p=cfg.flip_prob))
        transform.append(T.RandomVerticalFlip(p=cfg.flip_prob))

        if cfg.color_jitter_prob > 0.:
            transform.append(T.RandomApply(
                        [
                            T.ColorJitter(
                                **cfg.color_jitter
                            )
                        ],
                        p=cfg.color_jitter_prob,
                    ))

        # ugly unpack/tuple cast because T checks for list or tuple, but mine is Omega ListConfig
        if cfg.erase_iterations > 0:
            for _ in range(cfg.erase_iterations):
                scale = tuple(cfg.random_erasing.scale)
                ratio = tuple(cfg.random_erasing.ratio)
                transform.append(T.RandomErasing(scale=scale, ratio=ratio, p=cfg.random_erasing.p))
                
        if cfg.posterize:
            transform.append(T.RandomPosterize(**cfg.posterize))

        if cfg.elastic_prob > 0:
            # see-through-water 
            transform.append(
                T.RandomApply([
                    # should probably not change this 
                    T.ElasticTransform(
                        **cfg.elastic
                         # alpha=magnitude i.e. how much to displace
                         # sigma=smoothness i.e. size of the ripples. Should be about 10+ to keep a distorted galaxy with same morphology. Gets slower fast.
                    )],
                p=cfg.elastic_prob)
            )

        # tests show this is actually faster to do at the end, when using affine transform
        if cfg.pil_to_tensor:  # galaxydataset loads as PIL image, tensor is faster to work with
            transform.append(T.PILToTensor())

        # finally, shift to 0-1 float32 before hitting model etc
        transform.append(T.ToDtype(torch.float32, scale=True))
        # transform.append(T.ConvertImageDtype(torch.float32))

        # and optionally use timm cfg to normalize
        if cfg.normalize:
            transform.append(T.Normalize(**cfg.normalize))

        self.transform = T.Compose(transform)


    def __call__(self, image) -> torch.Tensor:

        return self.transform(image)


def default_view_config():
     return DictConfig(dict(
        pil_to_tensor=True,
        fixed_crop=False,
        output_size=224,
        greyscale=False,
        interpolation_method='bilinear',
        random_affine=dict(degrees=90, translate=(0.1, 0.1), scale=(1.2, 1.4), shear=(0,20,0,20)),
        random_perspective=False,
        center_crop=False,
        random_resized_crop=False,
        flip_prob=0.5,
        rotation_prob=0.,
        color_jitter_prob=0.,
        # color_jitter=dict(
        #     brightness=.5,
        #     contrast=.3,
        #     saturation=.5,
        #     hue=None
        # ),
        erase_iterations=5,
        random_erasing=dict(p=1., scale=[0.002, 0.007], ratio=[0.5, 2.]),
        posterize=False,
        elastic_prob=0.,
        normalize=False  # {mean, std} from timm cfg 'mean' and 'std'
        # elastic=dict(alpha=100, sigma=10.)
    ))

def minimal_view_config():
    return DictConfig(dict(
        pil_to_tensor=True,
        fixed_crop=False,
        output_size=224,
        greyscale=False,
        interpolation_method='bilinear',
        # no rotation, no translate, less aggressive crop, no shear
        random_affine=dict(degrees=0, translate=None, scale=(1.2, 1.2), shear=None),
        random_perspective=False,
        center_crop=False,
        random_resized_crop=False,
        flip_prob=0.,
        rotation_prob=0.,
        color_jitter_prob=0.,
        erase_iterations=0,
        posterize=False,
        elastic_prob=0.,
        normalize=False  # {mean, std} from timm cfg 'mean' and 'std'
    ))

def fast_view_config():
    return DictConfig(dict(
        pil_to_tensor=True,
        fixed_crop=False,
        output_size=224,
        greyscale=False,
        interpolation_method='bilinear',
        random_affine=False,
        random_perspective=False,
        center_crop=True,  # only transform
        random_resized_crop=False,
        flip_prob=0.,
        rotation_prob=0.,
        color_jitter_prob=0.,
        erase_iterations=0,
        posterize=False,
        elastic_prob=0.,
        normalize=False 
    ))

# for now, will deprecate
from galaxy_datasets.transforms_albumentations import minimal_transforms, fast_transforms, default_transforms, base_transforms
