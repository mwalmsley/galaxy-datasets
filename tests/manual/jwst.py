from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms as T

from galaxy_datasets.transforms_custom_torchvision import FixedCrop
from galaxy_datasets.transforms import default_view_config, GalaxyViewTransform


def test_direct_loading(png, jpg):
    for img in [png, jpg]:
        print(img.size)

        transforms = [
            T.PILToTensor(),
            FixedCrop(30, 30, 750, 750)
        ]
        transform = T.Compose(transforms)
        img = transform(img)
        print(img.shape)

        plt.imshow(img.permute(1, 2, 0))
        plt.axis('off')
        plt.show()


def test_transform_config(png, jpg):
    # test transform config
    transform_config = default_view_config()
    # transform_config.erase_iterations = 0  # optional
    transform_config.fixed_crop = {
        'lower_left_x': 30,
        'lower_left_y': 30,
        'upper_right_x': 750,
        'upper_right_y': 750
    }

    transform = GalaxyViewTransform(transform_config)
    for img in [png, jpg]:
        print(img.size)

        fig, axes = plt.subplots(2, 2, figsize=(10, 10))
        axes = axes.flatten()

        for n in range(4):
            img = transform(img)
            axes[n].imshow(img.permute(1, 2, 0))
            axes[n].axis('off')
        
        print(img.shape)

        plt.show()
  

if __name__ == "__main__":



    jpg_loc = '/home/walml/repos/galaxy-datasets/tests/data/jwst_grid/1_A1_composite.jpg'
    png_loc = '/home/walml/repos/galaxy-datasets/tests/data/jwst_grid/1_A1-composite.png'
    png =  Image.open(png_loc)
    jpg = Image.open(jpg_loc)

    # direct use
    # test_direct_loading(png, jpg)

    # test transform config
    test_transform_config(png, jpg)