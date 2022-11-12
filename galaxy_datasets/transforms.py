import albumentations as A
from albumentations.pytorch import ToTensorV2


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
