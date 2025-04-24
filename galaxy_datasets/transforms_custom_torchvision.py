# example transforms used for custom/specialized datasets
# e.g. handling Zooniverse grid subjects for GZ JWST COSMOS

import torch
import torchvision.transforms.v2 as T


class FixedCrop(torch.nn.Module):

    def __init__(self, lower_left_x, lower_left_y, upper_right_x, upper_right_y):
        super().__init__()
        # PIL format crop
        self.lower_left_x = lower_left_x
        self.lower_left_y = lower_left_y
        self.upper_right_x = upper_right_x
        self.upper_right_y = upper_right_y
        # pretty ugly, but I'm actually not sure how to only slice the last two dimensions without doing a reshape
        self.crop_hw = T.Lambda(lambda img: img[self.lower_left_y:self.upper_right_y, self.lower_left_x:self.upper_right_x])
        self.crop_chw = T.Lambda(lambda img: img[:, self.lower_left_y:self.upper_right_y, self.lower_left_x:self.upper_right_x])
        self.crop_nchw = T.Lambda(lambda img: img[:, :, self.lower_left_y:self.upper_right_y, self.lower_left_x:self.upper_right_x])

    def forward(self, img):
        if len(img.shape) == 2:
            # grayscale
            return self.crop_hw(img)    
        elif len(img.shape) == 3:
            # RGB
            return self.crop_chw(img)
        elif len(img.shape) == 4:
            # NCHW
            return self.crop_nchw(img)
        else:
            raise ValueError(f'Expected 2D, 3D or 4D tensor, got {len(img.shape)}D tensor instead. Shape: {img.shape}')

if __name__ == '__main__':

    from PIL import Image
    import matplotlib.pyplot as plt

    loc = '/home/walml/repos/galaxy-datasets/tests/examples/1_A1-composite.png'
    img = Image.open(loc)

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