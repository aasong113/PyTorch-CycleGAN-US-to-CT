from PIL import Image
import os
from torch.utils.data import Dataset
import numpy as np
import torch

class HorseZebraDataset(Dataset):
    def __init__(self, root_zebra, root_horse, transform=None):
        self.root_zebra = root_zebra
        self.root_horse = root_horse
        self.transform = transform

        self.zebra_images = os.listdir(root_zebra)
        #print(self.zebra_images)
        self.horse_images = os.listdir(root_horse)

        # assumes that we dont have paired datasets, where the length of the datasets are not equal. this is irrelevant for us.
        self.length_dataset = max(len(self.zebra_images), len(self.horse_images)) # 1000, 1500
        self.zebra_len = len(self.zebra_images)
        self.horse_len = len(self.horse_images)

    def __len__(self):
        return self.length_dataset

    def __getitem__(self, index):
        zebra_img = self.zebra_images[index % self.zebra_len]
        horse_img = self.horse_images[index % self.horse_len]

        # Get image path
        zebra_path = os.path.join(self.root_zebra, zebra_img)
        horse_path = os.path.join(self.root_horse, horse_img)

        # PIL open image for Torch Transforms.
        # zebra_img = Image.open(zebra_path)
        # horse_img = Image.open(horse_path)

        # For Albumentations Transforms..
        zebra_img = np.array(Image.open(zebra_path))
        horse_img = np.array(Image.open(horse_path))

        # apply transforms to the images.
        if self.transform:
            # For Torch Transforms
            # zebra_img = self.transform(zebra_img)
            # horse_img = self.transform(horse_img)
            # print(zebra_img.shape)

            # For Albumentations Transforms.
            augmentations = self.transform(image=zebra_img, image0=horse_img)
            zebra_img = augmentations["image"]
            horse_img = augmentations["image0"]

            # required for albumentations to work for proper size tensors. No using ToTensorV2, that shit not working.
            zebra_img = torch.from_numpy(zebra_img)
            zebra_img = zebra_img[None, :, :]
            horse_img = torch.from_numpy(horse_img)
            horse_img = horse_img[None, :, :]

        return zebra_img, horse_img





