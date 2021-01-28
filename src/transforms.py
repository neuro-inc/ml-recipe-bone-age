import torch
import cv2
import numpy as np
import logging
from torchvision import transforms
from albumentations import (
    ShiftScaleRotate, RandomGamma, Compose
)

logger = logging.getLogger(__file__)


class Crop(object):
    """Crop the image.

    Args:
        center (tuple): Center of the crop.
        crop_size (tuple): Desired output size.
    """

    def __init__(self, crop_center, crop_size):
        assert isinstance(crop_center, tuple)
        assert isinstance(crop_size, tuple)
        assert len(crop_center) == 2
        assert len(crop_size) == 2
        self.crop_center = crop_center
        self.crop_size = crop_size

    def __call__(self, sample):
        image = sample['image']

        top = self.crop_center[0] - self.crop_size[0] // 2
        left = self.crop_center[1] - self.crop_size[1] // 2
        image = image[top: top + self.crop_size[0],
                left: left + self.crop_size[1]]
        sample.update({'image': image})
        return sample


class Scale(object):
    """Scale the image.

    Args:
        scale (float): Scale factor
    """

    def __init__(self, scale):
        self.scale = scale

    def __call__(self, sample):
        image = sample['image']
        image = cv2.resize(image, dsize=None, fx=self.scale, fy=self.scale,
                           interpolation=cv2.INTER_CUBIC)
        sample.update({'image': image})
        return sample


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        if len(label.shape) == 0:
            label = np.asarray([label])

        label = np.expand_dims(label, axis=1).astype(np.float32)

        # numpy image: H x W x C, torch image: C x H x W
        image = np.expand_dims(image, axis=0)
        # standardize
        image = ((image - 127.5) / 127.5).astype(np.float32)

        # torch.from_numpy: The Torch Tensor and NumPy array will share their underlying memory (on CPU)
        sample.update({
            'image': torch.tensor(image),
            'label': torch.tensor(label)})
        return sample


class Augmentation(object):
    """Augment the image."""

    def __init__(self):
        self.aug = Compose([
            ShiftScaleRotate(p=1, shift_limit=0.05, scale_limit=0.05, rotate_limit=5,
                             border_mode=cv2.BORDER_CONSTANT),
            RandomGamma(p=1, gamma_limit=(80, 120)),
            #             RandomBrightnessContrast(p=1, brightness_limit=0.3, contrast_limit=0.3),
        ])

    def __call__(self, sample):
        image = sample['image']
        image = self.aug(image=image)['image']
        sample.update({'image': image})
        return sample


def get_transform(augmentation=True, crop_dict=None, scale=None):
    transform_list = []
    if augmentation:
        transform_list.append(Augmentation())
    if crop_dict:
        transform_list.append(Crop(crop_center=crop_dict['crop_center'], crop_size=crop_dict['crop_size']))
    if scale:
        transform_list.append(Scale(scale))
    transform_list.append(ToTensor())
    transform = transforms.Compose(transform_list)
    return transform
