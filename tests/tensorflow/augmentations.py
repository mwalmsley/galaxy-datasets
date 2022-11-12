
# from albumentations import Compose

# purely so we can import more easily from e.g. zoobot
# (needs only galaxy_datasets.tensorflow.augmentations)
from galaxy_datasets.transforms import default_transforms  

import matplotlib.pyplot as plt

from galaxy_datasets.tensorflow import galaxy_dataset
from galaxy_datasets.shared import tidal
from galaxy_datasets.transforms import default_transforms

from albumentations import (
    Compose, HorizontalFlip,
    Rotate
)

if __name__ == '__main__':


    catalog, label_cols = tidal.setup(
        root='/nvme1/scratch/walml/repos/pytorch-galaxy-datasets/roots/tidal',
        train=True,
        download=False)

    dataset = galaxy_dataset.get_image_dataset(
        image_paths=catalog['file_loc'],
        file_format='jpg',
        requested_img_size=64,
        labels=catalog['coarse_tidal_label'],
        greyscale=True,
        permute_channels=False,
        include_id_str=False
    )
   
    for images, labels in dataset.take(1):
        print(images.shape)
        print(labels)

    # transforms = Compose([
    #     Rotate(limit=90),
    #     HorizontalFlip()
    # ])
    transforms = default_transforms(
        greyscale=True,
        crop_scale_bounds=(0.7, 0.8),
        crop_ratio_bounds=(0.9, 1.1),
        resize_after_crop=48
    )
    # dataset = add_augmentations_to_dataset(dataset, transforms)

    # for image, label in dataset.take(1):
    #     print(image.shape)
    #     print(label)

    # # and then apply batch and then prefetch, as needed
    # dataset = dataset.shuffle(5000).batch(16).prefetch(buffer_size=AUTOTUNE)

    # for images, labels in dataset.take(1):
    #     print(images.shape)
    #     print(labels)

    # plt.imshow(images[0])
    # plt.show()


    n_repeats = 6
    n_galaxies = 4

    fig, rows = plt.subplots(ncols=n_galaxies, nrows=n_repeats)

    for n_repeat in range(n_repeats):

        for row in rows:
            augmented = add_transforms_to_dataset(dataset, transforms)

            augmented_subset = augmented.take(n_galaxies)
            for n_galaxy, (image, _) in enumerate(augmented_subset):
                row[n_galaxy].imshow(image)
                row[n_galaxy].axis('off')

    plt.tight_layout()
    plt.show()
