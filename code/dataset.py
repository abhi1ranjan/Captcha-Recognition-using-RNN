import albumentations
import torch

import numpy as np

from PIL import Image
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True   # to load truncated images


## This is a class for the dataset. 
## it is a clasfication problem, in which we have captcha images and you have to predict different images and characters in the image
class ClassificationDataset:
    def __init__(self, image_paths, targets, resize=None):    #image_paths = list of image paths, targets = list of targets, resize = (height, width)
        # resize = (height, width)
        self.image_paths = image_paths
        self.targets = targets
        self.resize = resize

        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
        self.aug = albumentations.Compose(
            [
                albumentations.Normalize(
                    mean, std, max_pixel_value=255.0, always_apply=True
                )
            ]
        )

    def __len__(self):       # returns the length of the dataset (number of images)
        return len(self.image_paths)

    def __getitem__(self, item):         # returns the image and the target, here item is the index of the image
        image = Image.open(self.image_paths[item]).convert("RGB")
        targets = self.targets[item]

        if self.resize is not None:    # resize the image, if resize is not None, resize is a tuple of (height, width), resample is the interpolation method
            image = image.resize(
                (self.resize[1], self.resize[0]), resample=Image.BILINEAR
            )

        image = np.array(image)      # convert the image to numpy array
        augmented = self.aug(image=image)   # apply the augmentations
        image = augmented["image"]
        # we are transposing the image because we want time dimension to correspond to the width of the image   - we are not doing this, different from akash tutorial
        image = np.transpose(image, (2, 0, 1)).astype(np.float32)    # convert the image to (channel, height, width) format, and convert the data type to float32

        return {
            "images": torch.tensor(image, dtype=torch.float),
            "targets": torch.tensor(targets, dtype=torch.long),     #targets is a list of 5 integers, so we convert it to torch.long. It was numpy array
        }
