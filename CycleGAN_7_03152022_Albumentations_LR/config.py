import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import torchvision.transforms as transforms


### HORSE = CT, ZEBRA = US !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TRAIN_DIR = "CT_US_data/train"
VAL_DIR = "CT_US_data/validation"
BATCH_SIZE = 1
LEARNING_RATE = 1e-05
LAMBDA_IDENTITY = 0.0
LAMBDA_CYCLE = 10
NUM_WORKERS = 4
NUM_EPOCHS = 100
ANIMAL_TYPE = "lamb"

# 1 for gray scale and 3 for RGB.
IMG_CHANNELS = 1

# Shuffle the indices of the data
SHUFFLE = True

# Save or Load model parameters
LOAD_MODEL = False
SAVE_MODEL = True
CHECKPOINT_GEN_H = "genh_CT_lr=" + str(LEARNING_RATE) + "batchsz_" + str(BATCH_SIZE) + "_epochs=" + str(NUM_EPOCHS) + "_shuffle=" + str(SHUFFLE) + "_" + ANIMAL_TYPE + ".pth.tar"
CHECKPOINT_GEN_Z = "genz_US_lr=" + str(LEARNING_RATE) + "batchsz_" + str(BATCH_SIZE) + "_epochs=" + str(NUM_EPOCHS) + "_shuffle=" + str(SHUFFLE) + "_" + ANIMAL_TYPE + ".pth.tar"
CHECKPOINT_CRITIC_H = "critich_CT_lr=" + str(LEARNING_RATE) + "batchsz_" + str(BATCH_SIZE) + "_epochs=" + str(NUM_EPOCHS) + "_shuffle=" + str(SHUFFLE) + "_" + ANIMAL_TYPE + ".pth.tar"
CHECKPOINT_CRITIC_Z = "criticz_US_lr=" + str(LEARNING_RATE) + "batchsz_" + str(BATCH_SIZE) + "_epochs=" + str(NUM_EPOCHS) + "_shuffle=" + str(SHUFFLE) + "_" + ANIMAL_TYPE + ".pth.tar"


### Torch Transforms
# transforms_train = transforms.Compose([
#     transforms.Resize((112, 152)),
#
#     transforms.ToTensor(),
#
#     transforms.RandomRotation([-15, 15]),
#
#     transforms.RandomHorizontalFlip(p=0.5),
#
#     transforms.Normalize([0.5], [0.5]),
#
#     ])
#
#
# transforms_val = transforms.Compose([
#     transforms.Resize((112, 152)),
#
#     transforms.ToTensor(),
#
#     transforms.Normalize([0.5], [0.5])]
# )

### Albumentation Transforms.
transforms_train = A.Compose(
    {
        # Canine Reshape
        #A.Resize(width=108, height=120),

        #Lamb Reshape
        A.Resize(width=112, height=152),
        #A.Resize(width=18, height=18),

        # Transforms we are interested in.
        A.HorizontalFlip(p=0.5),
        A.Rotate(limit=[-15, 15], border_mode=cv2.BORDER_REPLICATE),
        # The size of mean and std need to be changed according to RGB or Gray scale, for example RGB will be a
        # size 3 vector, where gray scale will be a size 1 vector.
        A.Normalize(mean=[0.5], std=[0.5], max_pixel_value=255),
        #ToTensorV2(),
    },
    additional_targets={"image0": "image"},
)

# For validation, we dont want any augmentations so we can compare to the ground truth.
transforms_val = A.Compose(
    {
        # Canine Reshape
        #A.Resize(width=108, height=120),

        #Lamb Reshape
        A.Resize(width=112, height=152),
        #A.Resize(width=18, height=18),

        # The size of mean and std need to be changed according to RGB or Gray scale, for example RGB will be a
        # size 3 vector, where gray scale will be a size 1 vector.
        A.Normalize(mean=[0.5], std=[0.5], max_pixel_value=255),
        #ToTensorV2(),
    },
    additional_targets={"image0": "image"},
)
