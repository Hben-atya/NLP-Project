import os
from torch.utils.data import Dataset
from PIL import Image
import torch
import torch.nn as nn
from torchvision import transforms
import pandas as pd


class Dataset_forMOCO(Dataset):

    def __init__(self, dataset_args, transform_flag=True, aug_transform=None, val=False):
        self.data_path = dataset_args['data_path_train']  # For get_item
        if val:
            self.data_path = dataset_args['data_path_val']
        self.aug_transform = aug_transform  # Transform function for images
        self.transform_flag = transform_flag  # Technically should always be True

    def __len__(self):
        return len(os.listdir(self.data_path))

    def __getitem__(self, item):
        # Path to image
        img_path = os.path.join(self.data_path, os.listdir(self.data_path)[item])
        # Get image
        img = Image.open(img_path).convert("L")
        # Apply transformation if exists (again, should technically exist)
        if self.transform_flag:
            # Resize image to (300,300) before augmenting for computational reasons
            resize = transforms.Resize((300, 300))
            # Pil to Tensor
            pil2tensor = transforms.PILToTensor()
            img_tensor = pil2tensor(img).float()
            img_tensor = resize(img_tensor)
            # If image is grayscale, make 3-channel grayscale where channel 1=2=3
            # if img_tensor.shape[0] == 1:
            #     img_tensor = img_tensor.repeat(3, 1, 1)
            # Apply random transformation twice, creating 2 different transforms of same image
            img1 = self.aug_transform(img_tensor)
            img2 = self.aug_transform(img_tensor)
        # Create pair of the two images
        data = {'image1': img1, 'image2': img2}
        return data


def data_augmentation(transform_args):
    """
    Transformations as described in MoCo original paper [Technical details section]
    :param transform_args:
    :return: aug_transform:
    """
    gaussian_blur_ = transform_args['GaussianBlur']  # Gaussian blur args
    # Sequential of: RandomResizedCrop, Normalization, RandomApply of
    # Gaussian Blur,Random Horizontal Flip with appropriate arguments
    aug_transform = nn.Sequential(
        transforms.RandomResizedCrop(transform_args['SizeCrop'], scale=(0.25, 1.0)),
        transforms.Normalize(mean=[0.5], std=[0.5]),
        transforms.RandomApply(torch.nn.ModuleList(
            [transforms.GaussianBlur(kernel_size=gaussian_blur_['kernel_size'],
                                     sigma=(gaussian_blur_['sigma_start'], gaussian_blur_['sigma_end']))
             ]), p=transform_args['p_apply']),

        transforms.RandomHorizontalFlip(p=transform_args['RandomHorizontalFlip']),
    )

    return aug_transform
