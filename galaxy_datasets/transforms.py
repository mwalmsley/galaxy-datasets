import albumentations as A
from albumentations.pytorch import ToTensorV2
from AstroAugmentations import image_domain


def default_transforms(crop_scale_bounds, crop_ratio_bounds, resize_after_crop, pytorch_greyscale=False):
    if pytorch_greyscale:
        transforms_to_apply = [A.Lambda(name='ToGray', image=ToGray(
            reduce_channels=True), always_apply=True)]
    else:
        transforms_to_apply = []

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
        # ToTensorV2() TODO may be needed with pytorch arg
    ]

    return A.Compose(transforms_to_apply)


def alternative_transforms(
    resize_size,
    shift_limit,
    scale_limit,
    rotate_limit,
    p_channelwise_dropout,
    p_elastic=0,
    elastic_alpha=1,
    elastic_sigma=1,
    p_flip=0.5,
    channelwise_dropout_max_fraction=0.2,
    channelwise_dropout_min_width=10,
    channelwise_dropout_min_height=10,
    channelwise_dropout_max_holes=100,
    pytorch_greyscale=False,
):
    """
    Alternative augmentations. Sequentially:
    ElasticTransform (optional) from https://ieeexplore.ieee.org/document/1227801
    ChannelWiseDropout (good for Decals) from AstroAugmentations
    ShiftScaleRotate instead of RandomResizedCrop for more control
    Flip along x or y axis


    resize_size:int
        resolution
    shift_limit:float
        max relative shift factor. 0.3 can have the galaxy pretty much
        at the edge *without scaling*
    scale_limit:float
        max relative max zoom *in* factor.
    rotate_limit:float
        max rotate angle in degrees.
    elastic_alpha:float
        how much to scale the random field
    elastic_sigma:float
        std. of the Gauss. kernel blurring the random field
    channelwise_dropout_max_fraction:float
        max area fraction to be affected
    channelwise_dropout_min_width:int
        minimum width of the hole in pixels
    channelwise_dropout_min_height:int
        minimum height of the hole in pixels
    """
    if pytorch_greyscale:
        transforms_to_apply = [
            A.Lambda(
                name="ToGray", image=ToGray(reduce_channels=True), always_apply=True
            )
        ]
    else:
        transforms_to_apply = []
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
                min_width=channelwise_dropout_min_width,
                min_height=channelwise_dropout_min_height,
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
        A.Flip(p=p_flip),
        ToTensorV2(),
    ]

    return A.Compose(transforms_to_apply)


# albumentations versuib of GrayscaleUnweighted
class ToGray():
    # will do nothing if already greyscale

    def __init__(self, reduce_channels=False):
        if reduce_channels:
            self.mean = lambda arr: arr.mean(axis=2, keepdims=True)
        else:
            self.mean = lambda arr: arr.mean(
                axis=2, keepdims=True).repeat(3, axis=2)

    def __call__(self, image, **kwargs):
        return self.mean(image)
