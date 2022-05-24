This is a CycleGAN model adapted from Aladdin Persson.

Code: 
    
    Generator Model: generator_model.py
    Discriminator Model: discriminator_model.py
    Configuration and Parameters: config.py
    Dataset Loader: dataset.py
    Model Saving: utils.py
    Training the model: train.py
    To test Image Transforms: image_transform_test.py

It is used to synthesize US from CT and CT to US from a Lamb spinal cord data published from IMPACT research group.
This experiment was done like so:


In config.py
Current Parameters: 
    
    100 epochs
    Learning rate: 1e-5
    Batch size = 1
    Input is gray scale image with channels = 1.

Image Transforms: 

    Flip across the vertical image axis
    Rotate -15 to 15 degrees with border replication padding.
    Intensity Normalization. 
    Image Resize: for lamb images -> A.Resize(width=112, height=152),
        However you must resize the image in A.transforms of config.py to be in sync with the size of the input image.
        Comment out to either use albumentations or torch transforms. Albumentation augmentations produce better results with model. Fixed Bug, Now you can use albumentations as well. In config.py, A.Transform, need to omit the ToTensorV2, because that is not working. Then in the dataset.py, need to perform the toTensor through PyTorch. 



-Saves model parameters at the epoch which minimizes the validation generator loss. 

-Saves plots for validation and training loss, and a npy array for validation and generator loss.

Training and Validation Dataset files require this path: 

    train/CT
    train/US
    validation/CT1
    validation/US1


