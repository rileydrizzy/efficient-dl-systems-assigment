"""
dataset.py

This module defines the Carvana dataset class and provides functions to load the training data and visualize images.
The Carvana dataset is used for image segmentation tasks, where the goal is to segment cars from images.
The Carvana class inherits from the PyTorch Dataset class and implements the necessary methods to load and preprocess the data.

Classes:
- Carvana: A custom dataset class for the Carvana image segmentation dataset.

Functions:
- get_train_data(): Loads the training data and returns a DataLoader object for batching and shuffling the data.
- im_show(img_list): Visualizes a list of images and their corresponding masks using Matplotlib.
   
Author: Ladipo Ipadeola
Date: 06/06/2026
"""
import os
from os.path import isfile, join
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torch.utils.data.dataset import Dataset
from torchvision import transforms

# Bach size and image shape constants for data loading and preprocessing
BATCH_SIZE = 64
IMAGE_SHAPE = (256, 256)


class Carvana(Dataset):
    """
    Carvana dataset class for image segmentation tasks.

    Methods
    ------- 
    __init__(self, root, transform=None):
        Initializes the Carvana dataset with the specified root directory and transformation settings.
    load_images(path):
        Loads the image file paths from the specified directory.
    __getitem__(self, index):
        Retrieves the image and corresponding mask at the specified index, applies transformations if necessary, and returns them as tensors.
    __len__(self):
        Returns the total number of samples in the dataset.

    """

    def __init__(self, root, transform=None):
        """
        Initialize the Carvana dataset.

        Parameters
        ---------- 
        root: str
            The root directory where the dataset is stored.
        transform: bool 
            A flag indicating whether to apply transformations to the images and masks. Default is None.

        Returns
        -------
        None
        """
        super().__init__()
        self.root = os.path.expanduser(root)
        self.transform = transform
        (self.data_path, self.labels_path) = ([], [])

        def load_images(path):
            """
            Load image file paths from the specified directory.

            Parameters
            ----------
            path: str
                The directory from which to load image file paths.
            Returns
            -------
            list
                A sorted list of image file paths in the specified directory.

            """
            images_dir = [join(path, file) for file in os.listdir(
                path) if isfile(join(path, file))]
            images_dir.sort()
            return images_dir
        self.data_path = load_images(self.root + "/data/train")
        self.labels_path = load_images(self.root + "/data/train_masks")

    def __getitem__(self, index):
        img = Image.open(self.data_path[index])
        target = Image.open(self.labels_path[index])

        if self.transform is not None:
            img = self.transform(img)
            target = self.transform(target)
            target = (target > 0).float()
        return img, target

    def __len__(self):
        return len(self.data_path)


def get_train_data():
    """
    Load the training data and return a DataLoader object    

    """
    transform_pipeline = transforms.Compose(
        [transforms.Resize(IMAGE_SHAPE), transforms.ToTensor()])
    train_dataset = Carvana(root='.', transform=transform_pipeline)
    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset, batch_size=BATCH_SIZE, pin_memory=False, num_workers=0)

    return train_loader


def im_show(img_list):
    to_PIL = transforms.ToPILImage()
    if len(img_list) > 9:
        # TODO Convert to Loguru Exception
        raise Exception("len(img) must be smaller than 10")
    fig, axes = plt.subplots(len(img_list), 2, figsize=(16, 16))
    fig.tight_layout()

    for (idx, sample) in enumerate(img_list):
        axes[idx][0].imshow(np.array(to_PIL(sample[0])))
        axes[idx][1].imshow(np.array(to_PIL(sample[1])))
        for ax in axes[idx]:
            ax.get_xaxis().set_visble(False)
            ax.get_yaxis().set_visble(False)
    plt.show()
