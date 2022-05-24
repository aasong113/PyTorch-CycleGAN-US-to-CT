from PIL import Image
import os
from torch.utils.data import Dataset
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import config
import matplotlib.pyplot as plt
import torch
import torchvision.transforms as transforms

def test():
    CT_path = r"C:\Users\asong18\Desktop\GAN_Synthesis\PyTorch_GAN\CycleGAN_8_03152022_adaptive_LR\CT_US_data_rot\train\CT\slice_0.tiff"
    US_path = r"C:\Users\asong18\Desktop\GAN_Synthesis\PyTorch_GAN\CycleGAN_8_03152022_adaptive_LR\CT_US_data_rot\train\US\slice_0.tiff"

    # Open np array image in PIL.
    imageCT = np.array(Image.open(CT_path))
    imageUS = np.array(Image.open(US_path))

    # Define the Transform.
    transform = A.Compose(
        {
            # Canine Reshape
            # A.Resize(width=108, height=120),

            # Lamb Reshape
            A.Resize(width=112, height=152),
            # A.Resize(width=18, height=18),

            # Transforms we are interested in.
            A.HorizontalFlip(p=1.0),
            #A.Rotate(limit=[-15, 15], border_mode=cv2.BORDER_REPLICATE),
            # The size of mean and std need to be changed according to RGB or Gray scale, for example RGB will be a
            # size 3 vector, where gray scale will be a size 1 vector.
            A.Normalize(mean=[0.5], std=[0.5], max_pixel_value=255),
            #ToTensorV2(),
        },
        additional_targets={"image0": "image"},
    )

    augmentations = transform(image=imageCT, image0=imageUS)
    CT_img = augmentations["image"]
    US_img = augmentations["image0"]

    plt.imshow(imageCT, cmap='gray')
    plt.title("Original CT")
    plt.show()

    plt.imshow(CT_img, cmap = 'gray')
    plt.title("Transformed CT")
    plt.show()

    plt.imshow(imageUS, cmap='gray')
    plt.title("Original US")
    plt.show()

    plt.imshow(US_img, cmap='gray')
    plt.title("Augmented US")
    plt.show()

    print(CT_img.shape)
    print(US_img.shape)


    # To tensor.
    z_t = torch.from_numpy(CT_img)
    z_t = z_t[None, :, :]
    h_t = torch.from_numpy(US_img)
    h_t = h_t[None, :, :]
    print(z_t.shape)
    print(h_t.shape)


    # transform = transforms.Compose([
    #     transforms.Resize((112, 152)),
    #
    #     transforms.ToTensor(),
    #
    #     transforms.RandomRotation([-45, 45]),
    #
    #     transforms.RandomHorizontalFlip(p=0.5),
    #
    #     transforms.Normalize([0.5], [0.5]),
    #
    # ])
    #
    #
    # transformed_image_1 = transform(image)
    #
    #
    # print("done")
    # print(transformed_image_1.shape)




if __name__ == "__main__":
    test()