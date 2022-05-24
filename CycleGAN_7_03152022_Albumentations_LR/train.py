import torch
from dataset import HorseZebraDataset
import os, sys
from utils import save_checkpoint, load_checkpoint
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import config
from tqdm import tqdm
from torchvision.utils import save_image
from discriminator_model import Discriminator
from generator_model import Generator
import matplotlib.pyplot as plt
import numpy as np

"""
Generate Validation Images and loss from the Network. 
"""
def validation_fn(disc_H, disc_Z, gen_Z, gen_H, val_loader,l1, mse, epoch_num, validation_images_dir):
    loop = tqdm(val_loader, leave=True)

    Discriminator_Loss = 0
    Generator_Loss = 0

    # This loop trains the model on the epoch data.
    for idx, (zebra, horse) in enumerate(loop):

        zebra = zebra.to(config.DEVICE)
        horse = horse.to(config.DEVICE)



        # Train Discriminators H and Z
        with torch.cuda.amp.autocast():
            fake_horse = gen_H(zebra)

            # Save fake horse.
            save_path_US = validation_images_dir+"/validation_US_epoch"+str(epoch_num)+"_idx"+str(idx)+".png"
            #print(save_path_US)
            save_image(fake_horse * 0.5 + 0.5, save_path_US)

            D_H_real = disc_H(horse)
            D_H_fake = disc_H(fake_horse.detach())

            # get the MSE. This compares the output of the discriminator and calculates the MSE between
            # the prediction that the real horse is real (1), and fake horse is fake (0).
            D_H_real_loss = mse(D_H_real, torch.ones_like(D_H_real))
            D_H_fake_loss = mse(D_H_fake, torch.zeros_like(D_H_fake))
            D_H_loss = D_H_real_loss + D_H_fake_loss

            # same idea as above.
            fake_zebra = gen_Z(horse)

            save_path_CT = validation_images_dir+"/validation_CT_epoch"+str(epoch_num)+"_idx"+str(idx)+".png"
            #print(save_path_CT)

            # save fake zebra
            save_image(fake_zebra * 0.5 + 0.5, save_path_CT)

            D_Z_real = disc_Z(zebra)
            D_Z_fake = disc_Z(fake_zebra.detach())

            # Uses MSE
            D_Z_real_loss = mse(D_Z_real, torch.ones_like(D_Z_real))
            D_Z_fake_loss = mse(D_Z_fake, torch.zeros_like(D_Z_fake))
            D_Z_loss = D_Z_real_loss + D_Z_fake_loss

            # Discriminator Loss: put it togethor
            D_loss = (D_H_loss + D_Z_loss) / 2

            # add the loss.
            Discriminator_Loss = Discriminator_Loss + D_loss.item()


        # Train Generators H and Z
        with torch.cuda.amp.autocast():
            # adversarial loss for both generators
            D_H_fake = disc_H(fake_horse)
            D_Z_fake = disc_Z(fake_zebra)
            loss_G_H = mse(D_H_fake, torch.ones_like(D_H_fake))
            loss_G_Z = mse(D_Z_fake, torch.ones_like(D_Z_fake))
            # print(loss_G_H)

            # cycle loss
            cycle_zebra = gen_Z(fake_horse)
            cycle_horse = gen_H(fake_zebra)

            # check to see the tensor shape for L1 loss.
            # print(cycle_horse.size())

            # Check L1 loss to make sure the sizes of the tensors are the same. Padding/stride in generator may mess this up.
            # this can be solved by reshaping the image in config.py -> transform -> resize
            cycle_zebra_loss = l1(zebra, cycle_zebra)
            cycle_horse_loss = l1(horse, cycle_horse)
            # print("here")

            # identity loss (remove these for efficiency if you set lambda_identity=0)
            identity_zebra = gen_Z(zebra)
            identity_horse = gen_H(horse)
            identity_zebra_loss = l1(zebra, identity_zebra)
            identity_horse_loss = l1(horse, identity_horse)

            # Generator Loss: add all togethor
            G_loss = (
                    loss_G_Z
                    + loss_G_H
                    + cycle_zebra_loss * config.LAMBDA_CYCLE
                    + cycle_horse_loss * config.LAMBDA_CYCLE
                    + identity_horse_loss * config.LAMBDA_IDENTITY
                    + identity_zebra_loss * config.LAMBDA_IDENTITY
            )

            Generator_Loss = Generator_Loss + G_loss.item()

    return Generator_Loss, Discriminator_Loss

"""
Train the network. 
Loss is calculated as a sum of all losses. 
"""
def train_fn(disc_H, disc_Z, gen_Z, gen_H, train_loader, opt_disc, opt_gen, l1, mse, d_scaler, g_scaler, epoch_num, train_images_dir):
    H_reals = 0
    H_fakes = 0
    loop = tqdm(train_loader, leave=True)

    Discriminator_Loss = 0
    Generator_Loss = 0
    # GAN_Loss = 0

    # This loop trains the model on the epoch data.
    for idx, (zebra, horse) in enumerate(loop):

        zebra = zebra.to(config.DEVICE)
        horse = horse.to(config.DEVICE)


        # check to see the tensor shape for L1 loss.
        # print(zebra.shape)

        # Train Discriminators H and Z
        with torch.cuda.amp.autocast():
            fake_horse = gen_H(zebra)
            D_H_real = disc_H(horse)
            D_H_fake = disc_H(fake_horse.detach())
            H_reals += D_H_real.mean().item()
            H_fakes += D_H_fake.mean().item()

            # get the MSE. This compares the output of the discriminator and calculates the MSE between
            # the prediction that the real horse is real (1), and fake horse is fake (0).
            D_H_real_loss = mse(D_H_real, torch.ones_like(D_H_real))
            D_H_fake_loss = mse(D_H_fake, torch.zeros_like(D_H_fake))
            D_H_loss = D_H_real_loss + D_H_fake_loss

            # same idea as above.
            fake_zebra = gen_Z(horse)
            D_Z_real = disc_Z(zebra)
            D_Z_fake = disc_Z(fake_zebra.detach())

            # Uses MSE
            D_Z_real_loss = mse(D_Z_real, torch.ones_like(D_Z_real))
            D_Z_fake_loss = mse(D_Z_fake, torch.zeros_like(D_Z_fake))
            D_Z_loss = D_Z_real_loss + D_Z_fake_loss

            # Discriminator Loss: put it togethor
            D_loss = (D_H_loss + D_Z_loss) / 2

            # add the loss.
            Discriminator_Loss = Discriminator_Loss + D_loss.item()

        # clears old gradients from the last step (otherwise you’d just accumulate the gradients from all loss.backward() calls)
        opt_disc.zero_grad()

        # computes the derivative of the loss w.r.t. the parameters (or anything requiring gradients) using backpropagation.
        d_scaler.scale(D_loss).backward()

        # causes the optimizer to take a step based on the gradients of the parameters.
        d_scaler.step(opt_disc)
        d_scaler.update()

        # Train Generators H and Z
        with torch.cuda.amp.autocast():
            # adversarial loss for both generators
            D_H_fake = disc_H(fake_horse)
            D_Z_fake = disc_Z(fake_zebra)
            loss_G_H = mse(D_H_fake, torch.ones_like(D_H_fake))
            loss_G_Z = mse(D_Z_fake, torch.ones_like(D_Z_fake))
            # print(loss_G_H)

            # cycle loss
            cycle_zebra = gen_Z(fake_horse)
            cycle_horse = gen_H(fake_zebra)

            # check to see the tensor shape for L1 loss.
            # print(cycle_horse.size())

            # Check L1 loss to make sure the sizes of the tensors are the same. Padding/stride in generator may mess this up.
            # this can be solved by reshaping the image in config.py -> transform -> resize
            cycle_zebra_loss = l1(zebra, cycle_zebra)
            cycle_horse_loss = l1(horse, cycle_horse)
            # print("here")

            # identity loss (remove these for efficiency if you set lambda_identity=0)
            identity_zebra = gen_Z(zebra)
            identity_horse = gen_H(horse)
            identity_zebra_loss = l1(zebra, identity_zebra)
            identity_horse_loss = l1(horse, identity_horse)

            # Generator Loss: add all togethor
            G_loss = (
                    loss_G_Z
                    + loss_G_H
                    + cycle_zebra_loss * config.LAMBDA_CYCLE
                    + cycle_horse_loss * config.LAMBDA_CYCLE
                    + identity_horse_loss * config.LAMBDA_IDENTITY
                    + identity_zebra_loss * config.LAMBDA_IDENTITY
            )

            Generator_Loss = Generator_Loss + G_loss.item()

        opt_gen.zero_grad()
        g_scaler.scale(G_loss).backward()
        g_scaler.step(opt_gen)
        g_scaler.update()

        # This saves the image periodically to check what is going on.
        if idx % 10 == 0:
            save_image(fake_horse * 0.5 + 0.5, f"{train_images_dir}/training_US_epoch_{epoch_num}_idx{idx}.png")
            save_image(fake_zebra * 0.5 + 0.5, f"{train_images_dir}/training_CT_epoch_{epoch_num}_idx{idx}.png")

        loop.set_postfix(H_real=H_reals / (idx + 1), H_fake=H_fakes / (idx + 1))

    return Generator_Loss, Discriminator_Loss


# release cuda Cache.
def force_cudnn_initialization():
    s = 32
    dev = torch.device('cuda')
    torch.nn.functional.conv2d(torch.zeros(s, s, s, s, device=dev), torch.zeros(s, s, s, s, device=dev))


def main():
    # 1 for Gray scale, 3 for RGB.
    channels = config.IMG_CHANNELS


    learning_rates = [0.001, 0.0001, 0.00001]
    #learning_rates = [0.001]

    # force_cudnn_initialization()
    for i in range(len(learning_rates)):
        #set Learning rate from system input.
        config.LEARNING_RATE = float(learning_rates[i])
        print(config.LEARNING_RATE)

        # Create a folder for job array.
        train_images_dir = f'training_images_lr={config.LEARNING_RATE}'
        validation_images_dir = f'validation_images_lr={config.LEARNING_RATE}'
        graph_images_dir = f'graph_images_lr={config.LEARNING_RATE}'
        dir_list = [train_images_dir, validation_images_dir, graph_images_dir]
        print(dir_list)
        # checking whether folder exists or not
        for i in range(len(dir_list)):
            if os.path.exists(dir_list[i]):

                # checking whether the folder is empty or not
                if len(os.listdir(dir_list[i])) == 0:
                    # removing the file using the os.remove() method
                    os.rmdir(dir_list[i])
                    os.mkdir(dir_list[i], 0o755)
                else:
                    # messaging saying folder not empty
                    print("Folder is not empty")
            else:
                os.mkdir(dir_list[i], 0o755)




        # Discriminator for classifying real or fake horses.
        disc_H = Discriminator(in_channels=channels).to(config.DEVICE)

        # discriminator for classifying real or fake zebras
        disc_Z = Discriminator(in_channels=channels).to(config.DEVICE)
        gen_Z = Generator(img_channels=channels, num_residuals=9).to(config.DEVICE)
        gen_H = Generator(img_channels=channels, num_residuals=9).to(config.DEVICE)

        # optimize our discriminator.
        opt_disc = optim.Adam(
            list(disc_H.parameters()) + list(disc_Z.parameters()),
            lr=config.LEARNING_RATE,
            betas=(0.5, 0.999),
        )

        # optimize our generator.
        opt_gen = optim.Adam(
            list(gen_Z.parameters()) + list(gen_H.parameters()),
            lr=config.LEARNING_RATE,
            betas=(0.5, 0.999),
        )

        # L1 loss for cycle consistency and identity loss.
        L1 = nn.L1Loss()

        # MSE loss for adversarial loss.
        mse = nn.MSELoss()

        if config.LOAD_MODEL:
            load_checkpoint(
                config.CHECKPOINT_GEN_H, gen_H, opt_gen, config.LEARNING_RATE,
            )
            load_checkpoint(
                config.CHECKPOINT_GEN_Z, gen_Z, opt_gen, config.LEARNING_RATE,
            )
            load_checkpoint(
                config.CHECKPOINT_CRITIC_H, disc_H, opt_disc, config.LEARNING_RATE,
            )
            load_checkpoint(
                config.CHECKPOINT_CRITIC_Z, disc_Z, opt_disc, config.LEARNING_RATE,
            )

        dataset = HorseZebraDataset(
            root_horse=config.TRAIN_DIR + "/CT", root_zebra=config.TRAIN_DIR + "/US", transform=config.transforms_train
        )
        val_dataset = HorseZebraDataset(
            root_horse=config.VAL_DIR + "/CT1", root_zebra=config.VAL_DIR + "/US1", transform=config.transforms_val
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=1,
            # Do not shuffle so we can see the images compared to ground truth.
            shuffle=False,
            pin_memory=True,
        )
        train_loader = DataLoader(
            dataset,
            batch_size=config.BATCH_SIZE,

            # This determines whether we shuffle them
            shuffle=config.SHUFFLE,
            num_workers=config.NUM_WORKERS,
            pin_memory=True
        )
        g_scaler = torch.cuda.amp.GradScaler()
        d_scaler = torch.cuda.amp.GradScaler()

        # Training Losses for plots.
        gen_loss = []
        disc_loss = []

        # Validation Losses for plots.
        gen_val_loss = []
        disc_val_loss = []

        min_generator_val_loss = 100000

        # Change save model names so they depic what is going on in the current loop
        CHECKPOINT_GEN_H = "genh_CT_lr=" + str(config.LEARNING_RATE) + "batchsz_" + str(config.BATCH_SIZE) + "_epochs=" + str(
            config.NUM_EPOCHS) + "_shuffle=" + str(config.SHUFFLE) + "_" + config.ANIMAL_TYPE + ".pth.tar"
        CHECKPOINT_GEN_Z = "genz_US_lr=" + str(config.LEARNING_RATE) + "batchsz_" + str(config.BATCH_SIZE) + "_epochs=" + str(
            config.NUM_EPOCHS) + "_shuffle=" + str(config.SHUFFLE) + "_" + config.ANIMAL_TYPE + ".pth.tar"
        CHECKPOINT_CRITIC_H = "critich_CT_lr=" + str(config.LEARNING_RATE) + "batchsz_" + str(config.BATCH_SIZE) + "_epochs=" + str(
            config.NUM_EPOCHS) + "_shuffle=" + str(config.SHUFFLE) + "_" + config.ANIMAL_TYPE + ".pth.tar"
        CHECKPOINT_CRITIC_Z = "criticz_US_lr=" + str(config.LEARNING_RATE) + "batchsz_" + str(config.BATCH_SIZE) + "_epochs=" + str(
            config.NUM_EPOCHS) + "_shuffle=" + str(config.SHUFFLE) + "_" + config.ANIMAL_TYPE + ".pth.tar"

        for epoch in range(config.NUM_EPOCHS):

            g_train_loss, d_train_loss = train_fn(disc_H, disc_Z, gen_Z, gen_H, train_loader, opt_disc, opt_gen, L1, mse, d_scaler,
                                      g_scaler,
                                      epoch, train_images_dir)
            gen_loss.append(g_train_loss)
            disc_loss.append(d_train_loss)


            # Save Validation images and Loss.
            g_val_loss, d_val_loss = validation_fn(disc_H, disc_Z, gen_Z, gen_H, val_loader, L1, mse, epoch, validation_images_dir)
            gen_val_loss.append(g_val_loss)
            disc_val_loss.append(d_val_loss)


            # Save the model at current epoch if the generator loss is the minimum generator loss.
            if config.SAVE_MODEL and g_val_loss < min_generator_val_loss:
                min_generator_val_loss = g_val_loss
                save_checkpoint(gen_H, opt_gen, epoch, filename=config.CHECKPOINT_GEN_H)
                save_checkpoint(gen_Z, opt_gen, epoch, filename=config.CHECKPOINT_GEN_Z)
                save_checkpoint(disc_H, opt_disc, epoch, filename=config.CHECKPOINT_CRITIC_H)
                save_checkpoint(disc_Z, opt_disc, epoch, filename=config.CHECKPOINT_CRITIC_Z)



        # Save Training Generator and Discriminator Plots.
        fig, ax1 = plt.subplots()
        plt.title("Training Generator and Discriminator Loss vs. Epoch")
        ax2 = ax1.twinx()
        ax1.plot(range(config.NUM_EPOCHS), gen_loss,'g-',  label = "Generator Loss")
        ax1.set_ylabel('Generator Loss (L1 + MSE)', color = 'g')

        ax2.plot(range(config.NUM_EPOCHS), disc_loss,'b-', label = "Discriminator Loss")
        ax2.set_ylabel('Discriminator Loss (MSE)', color='b')
        fig.legend(loc = 3, bbox_to_anchor=(0.6,0.7))
        ax1.set_xlabel("Epochs")

        #print(f"{graph_images_dir}/gen_disc_training_loss_lr={config.LEARNING_RATE}_epochs={config.NUM_EPOCHS}_shuffle={config.SHUFFLE}_{config.ANIMAL_TYPE}_.png")

        plt.savefig(
            f'{graph_images_dir}/gen_disc_training_loss_lr={config.LEARNING_RATE}_epochs={config.NUM_EPOCHS}_shuffle={config.SHUFFLE}_{config.ANIMAL_TYPE}_.png')
        plt.show()

        # Save Validation Generator and Discriminator Plots.
        fig, ax1 = plt.subplots()
        plt.title("Validation Generator and Discriminator Loss vs. Epoch")
        ax2 = ax1.twinx()
        ax1.plot(range(config.NUM_EPOCHS), gen_val_loss, 'g-', label="Generator Loss")
        ax1.set_ylabel('Generator Loss (L1 + MSE)', color='g')

        ax2.plot(range(config.NUM_EPOCHS), disc_val_loss, 'b-', label="Discriminator Loss")
        ax2.set_ylabel('Discriminator Loss (MSE)', color='b')
        fig.legend(loc = 3, bbox_to_anchor=(0.6,0.7))
        ax1.set_xlabel("Epochs")
        plt.savefig(
            f'{graph_images_dir}/gen_disc_validation_loss_lr={config.LEARNING_RATE}_epochs={config.NUM_EPOCHS}_shuffle={config.SHUFFLE}_{config.ANIMAL_TYPE}_.png')
        plt.show()

        # Save Training Loss for Generator and Discriminator as .npy file.
        train_gen_loss = np.concatenate((np.array([range(config.NUM_EPOCHS)]), np.array([gen_loss])), axis = 0)
        np.save(f'{graph_images_dir}/gen_train_loss_data_lr={config.LEARNING_RATE}_epochs={config.NUM_EPOCHS}_shuffle={config.SHUFFLE}_{config.ANIMAL_TYPE}_.png', train_gen_loss)
        print(train_gen_loss)

        train_disc_loss = np.concatenate((np.array([range(config.NUM_EPOCHS)]), np.array([disc_loss])), axis = 0)
        np.save(f'{graph_images_dir}/disc_train_loss_data_lr={config.LEARNING_RATE}_epochs={config.NUM_EPOCHS}_shuffle={config.SHUFFLE}_{config.ANIMAL_TYPE}_.png', train_disc_loss)

        # Save Validation Loss for Generator and Discriminator as .npy file.
        val_gen_loss = np.concatenate((np.array([range(config.NUM_EPOCHS)]), np.array([gen_val_loss])), axis = 0)
        np.save(f'{graph_images_dir}/gen_val_loss_data_lr={config.LEARNING_RATE}_epochs={config.NUM_EPOCHS}_shuffle={config.SHUFFLE}_{config.ANIMAL_TYPE}_.png', val_gen_loss)

        val_disc_loss = np.concatenate((np.array([range(config.NUM_EPOCHS)]), np.array([disc_val_loss])), axis = 0)
        np.save(f'{graph_images_dir}/disc_val_loss_data_lr={config.LEARNING_RATE}_epochs={config.NUM_EPOCHS}_shuffle={config.SHUFFLE}_{config.ANIMAL_TYPE}_.png', val_disc_loss)


        print(config.LEARNING_RATE)
        #Individual Figures
        """
           # Plot loss curves and Save.
        plt.title("Generator Loss vs. Epoch")
        plt.plot(range(config.NUM_EPOCHS), gen_loss, label="Generator (L1 + MSE)")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(
            f"saved_images/gen_loss_plot_lr=" + str(config.LEARNING_RATE) + "_epochs=" + str(config.NUM_EPOCHS)
            + "_shuffle=" + str(config.SHUFFLE) + "_" + str(config.ANIMAL_TYPE) + "_.png")
        plt.show()
    
        # Plot loss curves and Save.
        plt.title("Discriminator Loss vs. Epoch")
        plt.plot(range(config.NUM_EPOCHS), disc_loss, label="Discriminator (MSE)")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(
            f"saved_images/disc_loss_plot_lr=" + str(config.LEARNING_RATE) + "_epochs=" + str(config.NUM_EPOCHS)
            + "_shuffle=" + str(config.SHUFFLE) + "_" + str(config.ANIMAL_TYPE) + "_.png")
        plt.show()
    
        """


if __name__ == "__main__":
    main()
