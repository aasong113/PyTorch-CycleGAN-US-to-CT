This is a CycleGAN model adapted from Aladdin Persson.
It is used to synthesize US from CT and CT to US from a Lamb spinal cord data published from IMPACT research group.
This experiment was done like so:

-50 epochs

-Flip across the axis that splits the image symmetrically.

-Rotate -15 to 15 degrees with border replication padding.
Most things can be changed in config.py

-However you must resize the image in A.transforms of config.py to be in sync with the size of the input image.
-Comment out to either use albumentations or torch transforms. Albumentation augmentations produce better results with 
model. Fixed Bug, Now you can use albumentations as well. In config.py, A.Transform, need to omit the ToTensorV2, 
because that is not working. Then in the dataset.py, need to perform the toTensor through PyTorch. 

-Input is gray scale image with channels = 1.

-Saves model parameters at the epoch which minimizes the validation generator loss. 

-Saves plots for validation and training loss, and a npy array for validation and generator loss.

Training and Validation Dataset files require this path: 

    train/CT
    train/US
    validation/CT1
    validation/US1


