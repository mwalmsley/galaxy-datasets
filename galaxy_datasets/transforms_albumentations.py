
# import numpy as np
# import albumentations as A
# from albumentations.pytorch import ToTensorV2

# """And below, the older transform options"""

# # deprecated in favour of torchvision and will soon be removed
# def default_transforms(
#     crop_scale_bounds=(0.7, 0.8),
#     crop_ratio_bounds=(0.9, 1.1),
#     initial_center_crop=None,
#     resize_after_crop=224, 
#     pytorch_greyscale=False,
#     to_float=True,  # set to True when loading images directly, False via webdatasets (which normalizes to 0-1 on decode)
#     to_tensor=True
#     ) -> A.Compose:

#     transforms_to_apply = base_transforms(pytorch_greyscale)

#     if initial_center_crop:
#         transforms_to_apply += [
#             A.CenterCrop(
#                 height=initial_center_crop,  # initial crop
#                 width=initial_center_crop,
#                 # always_apply=True
#             )
#         ]

#     transforms_to_apply += [
#         # A.ToFloat(),
#         # anything outside of the original image is set to 0.
#         A.Rotate(limit=180, interpolation=1,
#                     always_apply=True, border_mode=0, value=0),
#         A.RandomResizedCrop(
#             size=(resize_after_crop, resize_after_crop), # after crop resize
#             scale=crop_scale_bounds,  # crop factor
#             ratio=crop_ratio_bounds,  # crop aspect ratio
#             interpolation=1,  # This is "INTER_LINEAR" == BILINEAR interpolation. See: https://docs.opencv.org/3.4/da/d54/group__imgproc__transform.html
#             # always_apply=True
#         ),  # new aspect ratio
#         A.VerticalFlip(p=0.5)
#     ]
#     if to_float:
#         transforms_to_apply += [A.ToFloat(max_value=255.0, always_apply=True)]

#     if to_tensor:
#         transforms_to_apply += [ToTensorV2(always_apply=True)]

#     return A.Compose(transforms_to_apply)



# def minimal_transforms(
#     resize_after_crop=224, 
#     pytorch_greyscale=False
#     ) -> A.Compose:
#     # for testing how much augmentations slow training / picking CPU to allocate
#     transforms_to_apply = base_transforms(pytorch_greyscale)
#     transforms_to_apply += [
#         A.CenterCrop(
#             height=resize_after_crop,
#             width=resize_after_crop,
#             always_apply=True
#         )
#     ]
#     return A.Compose(transforms_to_apply)


# def fast_transforms(
#     pytorch_greyscale=False,
#     resize_after_crop=224
#     ):
#     # middle ground between default and minimal transforms
#     # faster than default because we avoid interpolation
#     # better than minimal because we have some rotation/flip, and use random (non-central) crop
#     # should only be used for proper training if you already resize the images
#     # such that the resize_after_crop FoV makes sense (as this is just cropping)
#     # for 0.75x=224, x=300, so save at 300x300 pixels!
#     assert resize_after_crop == 224  # check user isn't attempting to change this
#     transforms_to_apply = base_transforms(pytorch_greyscale)
#     transforms_to_apply += [
#     #     A.RandomCrop(
#     #         height=resize_after_crop,
#     #         width=resize_after_crop,
#     #         always_apply=True
#     #     ),
#         A.Flip(),
#         A.RandomRotate90()
#     ]
#     return A.Compose(transforms_to_apply)


# def base_transforms(pytorch_greyscale):
#     if pytorch_greyscale:
#         return [
#             A.Lambda(
#                 name="ToGray", image=ToGray(reduce_channels=True) #, always_apply=True
#             )
#         ]
#     else:
#        return [
#             A.Lambda(name="RemoveAlpha", image=RemoveAlpha()) #, always_apply=True)
#         ]
        


# # albumentations version of GrayscaleUnweighted
# class ToGray():

#     def __init__(self, reduce_channels=False):
#         self.reduce_channels = reduce_channels

#     def forward(self, img):
#         if len(img.shape) == 2:  # saved to disk as greyscale already, with no channel
#             img = np.expand_dims(img, axis=2) # add channel=1 dimension
#         # print(img.shape)
#         if self.reduce_channels:
#             return img.mean(axis=2, keepdims=True)
#         else:
#             return img.mean(axis=2, keepdims=True).repeat(3, axis=2)

#     def __call__(self, image, **kwargs):
#         return self.forward(image)

# class RemoveAlpha():

#     def __init__(self):
#         # some png images have fourth alpha channel with value of 255 everywhere (i.e. opaque). averaging over this adds incorrect offset
#         pass

#     def forward(self, img):
#         if img.shape[2] == 4:
#             return img[:, :, :3]
#         return img

#     def __call__(self, image, **kwargs):
#         return self.forward(image)



# # def interpolation_lookup(interpolation_str='bilinear'):
# #         # PIL.Image.bilinear
# #         return getattr(PIL.Image, interpolation_str.upper())
