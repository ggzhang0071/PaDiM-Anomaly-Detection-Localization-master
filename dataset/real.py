import cv2
import os
import numpy as np
import torch
import torch.utils.data
from common_lib import DataManager
import random
from data_loader import DataLoader
from .base import DatasetBase


class DatasetReal(DatasetBase):
    """ Dataset for json file """
    def __init__(self, json_file, image_data_root, transform, resampling=None):
        """
        Args:
            json_file: DataManager Json format
            image_data_root: root directory to images
        """
        super(DatasetReal, self).__init__(resampling)
        assert os.path.exists(json_file), "{} doesnot exist".format(json_file)
        self.resize_transform = transform

        self.data_loader = DataLoader(image_data_root=image_data_root)
        self.data = DataManager.from_json(json_file)

        self.initialize_dataset()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        rec = self.get_record(idx)
        try:
            dat = self.data_loader.load(rec)
            image1_ng, image1_ok = dat['image'], dat['template']
            image2_ok = image1_ok.copy()
            image2_ng = image1_ng.copy()

            # load loss mask
            loss_mask = 1. - dat['loss_mask']
        except:
            raise RuntimeError("Image loading fails: idx = {}".format(idx))

        source = self.resize_transform(image=image1_ok)['image'] /255
        """image1_ng = self.resize_transform(image=image1_ng)['image'] / 255.
        image2_ok = self.resize_transform(image=image2_ok)['image'] / 255.
        image2_ng = self.resize_transform(image=image2_ng)['image'] / 255.
        loss_mask = self.resize_transform(image=loss_mask)['image']

        source = np.dstack([image1_ok, image1_ng, image2_ok])
        target = np.dstack([image2_ng, loss_mask])

        # permute dimensions
        source = source.transpose(2, 0, 1).astype('float32')
        target = target.transpose(2, 0, 1).astype('float32')"""
        source = source.transpose(2, 0, 1).astype('float32')
        return source


# from dataset.real import DatasetReal as Dataset
# from config import default_config
# import matplotlib.pyplot as plt
# from tqdm import tqdm

# from albumentations.augmentations import transforms
# from albumentations.core.composition import Compose

# train_transform = Compose([
#     transforms.Resize(default_config.Input_Height, default_config.Input_Width),
# ])

# data = Dataset(
#     json_file='assets/toy.json',
#     image_data_root=default_config.Image_Data_Root,
#     transform=train_transform,
#     resampling='balance'
# )
# print(len(data))


# for source, target, info in tqdm(data):
#     source = source.transpose(1, 2, 0)
#     target = target.transpose(1, 2, 0)
#     image1_ok = source[...,:3]
#     image1_ng = source[...,3:6]
#     image2_ok = source[...,6:]
#     image2_ng = target[...,:3]
#     loss_mask = target[...,3:]
#     _, ax = plt.subplots(2,3)
#     ax[0][0].imshow(image1_ok)
#     ax[0][1].imshow(image1_ng)
#     ax[1][0].imshow(image2_ok)
#     ax[1][1].imshow(image2_ng)
#     ax[0][2].imshow(loss_mask,   vmin=0, vmax=1)
#     ax[1][2].imshow(1-loss_mask, vmin=0, vmax=1)
#     plt.show()
#     pass

