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
    zebra_path = r"C:\Users\asong18\Desktop\GAN_Synthesis\PyTorch_GAN\CycleGAN_6_03152022_Albumentations_test\CT_US_canine_downsampled_partial\train\CT\slice_0.png"

    #image = np.array(Image.open(zebra_path).convert("RGB"))
    # horse_img = np.array(Image.open(horse_path).convert("RGB"))

    # import grayscale
    # image = np.expand_dims(np.array(Image.open(zebra_path)), axis=2)
    #image = Image.open(zebra_path)
    image = np.array(Image.open(zebra_path))
    print(image.shape)
    #image = torch.from_numpy(image.numpy()[:, :, ::-1].copy())

    #image = np.eye(100)
    #image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    print(isinstance(image, np.ndarray))

    #print(image.dtype)

    #print(image)
    print(np.amax(image))
    print(np.amin(image))

    # image = cv2.imread(zebra_path, cv2.IMREAD_UNCHANGED)
    # print(image.dtype)
    # print(image.shape)
    # print(image)
    transform = A.Compose(
        {
            # Canine Reshape
            # A.Resize(width=108, height=120),

            # Lamb Reshape
            A.Resize(width=152, height=112),
            # A.Resize(width=18, height=18),

            # Transforms we are interested in.
            A.HorizontalFlip(p=1.0),
            A.Rotate(limit=[-15, 15], border_mode=cv2.BORDER_REPLICATE),
            # The size of mean and std need to be changed according to RGB or Gray scale, for example RGB will be a
            # size 3 vector, where gray scale will be a size 1 vector.
            A.Normalize(mean=[0.5], std=[0.5], max_pixel_value=255),
            #ToTensorV2(),
        },
        additional_targets={"image0": "image"},
    )

    augmentations = transform(image=image, image0=image)
    zebra_img = augmentations["image"]
    horse_img = augmentations["image0"]

    print(zebra_img.shape)
    print(horse_img.shape)

    z_t = torch.from_numpy(zebra_img)
    z_t = z_t[None, :, :]
    h_t = torch.from_numpy(horse_img)
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
    plt.imshow(zebra_img, cmap = 'gray')
    plt.show()
    plt.imshow(horse_img, cmap='gray')
    plt.show()




if __name__ == "__main__":
    test()