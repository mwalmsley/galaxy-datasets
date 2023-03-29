import typing

import albumentations as A


def default_transforms(
    crop_scale_bounds=(0.7, 0.8),
    crop_ratio_bounds=(0.9, 1.1),
    resize_after_crop=224, 
    pytorch_greyscale=False
    ) -> typing.Dict[str, typing.Any]:

    transforms_to_apply = [
        A.Lambda(name="RemoveAlpha", image=RemoveAlpha(), always_apply=True)
    ]

    if pytorch_greyscale:
        transforms_to_apply += [
            A.Lambda(
                name="ToGray", image=ToGray(reduce_channels=True), always_apply=True
            )
        ]

    transforms_to_apply += [
        # A.ToFloat(),
        # anything outside of the original image is set to 0.
        A.Rotate(limit=180, interpolation=1,
                    always_apply=True, border_mode=0, value=0),
        A.RandomResizedCrop(
            height=resize_after_crop,  # after crop resize
            width=resize_after_crop,
            scale=crop_scale_bounds,  # crop factor
            ratio=crop_ratio_bounds,  # crop aspect ratio
            interpolation=1,  # This is "INTER_LINEAR" == BILINEAR interpolation. See: https://docs.opencv.org/3.4/da/d54/group__imgproc__transform.html
            always_apply=True
        ),  # new aspect ratio
        A.VerticalFlip(p=0.5),
    ]

    return A.Compose(transforms_to_apply)


def astroaugmentation_transforms(
    resize_size: int,
    shift_limit: float,
    scale_limit: float,
    rotate_limit: float,
    p_channelwise_dropout: float,
    p_elastic=0,
    elastic_alpha=1,
    elastic_sigma=1,
    channelwise_dropout_max_fraction=0.2,
    channelwise_dropout_min_length=10,
    channelwise_dropout_max_holes=100,
    pytorch_greyscale=False,
) -> typing.Dict[str, typing.Any]:
    """
    Astrophysically-inspired image transforms from AstroAugmentations (Bowles in prep.)
    Intended to mimic common distortions to optical telescope images.
    
    Sequentially:
    - ElasticTransform (optional) from https://ieeexplore.ieee.org/document/1227801
    - ChannelWiseDropout (good for Decals) from AstroAugmentations
    - ShiftScaleRotate instead of RandomResizedCrop for more control
    - Flip along x or y axis

    Args:
        resize_size (int): resolution
        shift_limit (float): max relative shift factor. 0.3 can have the galaxy pretty much
                            at the edge *without scaling*
        scale_limit (float):  max relative max zoom *in* factor
        rotate_limit (float): max rotate angle in degrees
        p_channelwise_dropout (float): _description_
        p_elastic (int, optional): how much to scale the random field. Defaults to 0.
        elastic_alpha (int, optional): std. of the Gauss. kernel blurring the random field. Defaults to 1.
        elastic_sigma (int, optional):  max area fraction to be affected. Defaults to 1.
        channelwise_dropout_max_fraction (float, optional): _description_. Defaults to 0.2.
        channelwise_dropout_min_length (int, optional):  minimum length of the hole in pixels. Defaults to 10. May be horizontal or vertical (width or height)
        channelwise_dropout_max_holes (int, optional): _description_. Defaults to 100.
        pytorch_greyscale (bool, optional): _description_. Defaults to False.

    Returns:
        typing.Dict[str, typing.Any]: _description_
    """
    # wrapped in Try/Except to avoid making AstroAugmentations a package requirement for only this optional transform
    try:
        from AstroAugmentations import image_domain
    except ImportError:
        raise ImportError(
            'Trying to use astroaugmentation_transforms but AstroAugmentations is not installed\n \
            Please install via git. See instructions at https://github.com/mb010/AstroAugmentations#quick-start'
        )
    from albumentations.pytorch import ToTensorV2  # also required pytorch

    transforms_to_apply = [
        A.Lambda(name="RemoveAlpha", image=RemoveAlpha(), always_apply=True)
    ]

    if pytorch_greyscale:
        transforms_to_apply += [
            A.Lambda(
                name="ToGray", image=ToGray(reduce_channels=True), always_apply=True
            )
        ]

    transforms_to_apply += [
        A.LongestMaxSize(max_size=resize_size),
        A.ElasticTransform(
            alpha=elastic_alpha,
            sigma=elastic_sigma,
            alpha_affine=0,
            interpolation=1,
            border_mode=1,
            value=0,
            p=p_elastic,
        ),
        A.Lambda(
            name="MissingData",
            image=image_domain.optical.ChannelWiseDropout(
                max_fraction=channelwise_dropout_max_fraction,
                # max is up to channelwise_dropout_max_fraction
                min_width=channelwise_dropout_min_length,
                min_height=channelwise_dropout_min_length,
                max_holes=channelwise_dropout_max_holes,
                channelwise_application=True,
            ),
            p=p_channelwise_dropout,
        ),
        A.ShiftScaleRotate(
            shift_limit=shift_limit,
            scale_limit=(0, scale_limit),
            rotate_limit=rotate_limit,
            interpolation=2,
            border_mode=0,
            p=1,
        ),
        # TODO maybe add sersic/gaussian
        A.Flip(p=0.5),
        ToTensorV2(),
    ]

    return A.Compose(transforms_to_apply)


# albumentations version of GrayscaleUnweighted
class ToGray():

    def __init__(self, reduce_channels=False):
        self.reduce_channels = reduce_channels

    def forward(self, img):
        if self.reduce_channels:
            return img.mean(axis=2, keepdims=True)
        else:
            return img.mean(axis=2, keepdims=True).repeat(3, axis=2)

    def __call__(self, image, **kwargs):
        return self.forward(image)

class RemoveAlpha():

    def __init__(self):
        # some png images have fourth alpha channel with value of 255 everywhere (i.e. opaque). averaging over this adds incorrect offset
        pass

    def forward(self, img):
        return img[:, :, :3]

    def __call__(self, image, **kwargs):
        return self.forward(image)
