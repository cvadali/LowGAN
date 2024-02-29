import torch
import torchio as tio
import torch.nn as nn
import torch.nn.functional as F
from pytorch_msssim import ssim
from torch.optim.lr_scheduler import MultiStepLR
from unet import UNet
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import multiprocessing
import gc
import random
from pathlib import Path
import glob
import argparse

# prepare batch for training
def prepare_batch(batch_in, device):
  recon = batch_in['recon'][tio.DATA].to(device)
  hifi = batch_in['hifi'][tio.DATA].to(device)
  return recon, hifi

# create patches for training set
def create_patches_training_set(images_dir, hifi_and_mask_dir, training_batch_size=8, max_queue_length=300, 
                                samples_per_volume=80, sampler=tio.data.WeightedSampler((32,32,32), 'sampling_map'), 
                                num_workers=multiprocessing.cpu_count(), shuffle_subjects=True, shuffle_patches=True):
    
    # axial, coronal, sagittal images
    image_paths_coronal = sorted(images_dir.glob('*_recon_*_coronal.nii.gz'))
    image_paths_axial = sorted(images_dir.glob('*_recon_*_axial.nii.gz'))
    image_paths_sagittal = sorted(images_dir.glob('*_recon_*_sagittal.nii.gz'))

    # real hifi images and brainmask
    image_paths_hifi = sorted(hifi_and_mask_dir.glob('*_hifi_*_padded.nii.gz'))
    image_paths_mask = sorted(hifi_and_mask_dir.glob('*_hifi_brainmask.nii.gz'))

    training_subjects_4d = []
    for (coronal_path, axial_path, sagittal_path, hifi_path, mask_path) in zip(image_paths_coronal, image_paths_axial, image_paths_sagittal, image_paths_hifi, image_paths_mask):
        
        subject = tio.Subject(
            recon=tio.ScalarImage([coronal_path,axial_path,sagittal_path]),
            hifi=tio.ScalarImage(hifi_path),
            sampling_map=tio.Image(mask_path, type=tio.SAMPLING_MAP)

        )
        training_subjects_4d.append(subject)

    training_transform = tio.Compose([tio.ZNormalization(masking_method=tio.ZNormalization.mean), tio.Resample('recon')])

    training_set = tio.SubjectsDataset(training_subjects_4d, transform=training_transform)

    print('Training set:', len(training_set), 'subjects')

    # create training set
    patches_training_set = tio.Queue(
        subjects_dataset=training_set,
        max_length=max_queue_length,
        samples_per_volume=samples_per_volume,
        sampler=sampler,
        num_workers=num_workers,
        shuffle_subjects=shuffle_subjects,
        shuffle_patches=shuffle_patches
        )

    # create training data loader
    training_loader_patches = torch.utils.data.DataLoader(patches_training_set, batch_size=training_batch_size)

    return training_loader_patches

# SSIM L1 loss function
def ssim_l1_loss(img_in, img_out, device):
    l1 = nn.L1Loss().to(device)

    loss = (1-ssim(img_in, img_out).to(device))*0.5 + 0.5*(l1(img_in,img_out))
    return loss

# Simple 3D UNet model
def get_3dunet_model_and_optimizer(device):
    # using the UNet package, we generate a 2D-UNet in PyTorch
    model = UNet(
        in_channels=3,
        out_classes=1,
        dimensions=3,
        num_encoding_blocks=5,
        out_channels_first_layer=64,
        normalization='batch',
        upsampling_type='linear',
        padding=True,
        activation='PReLU',
        residual=True,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters())
    return model, optimizer

# train the 3D UNet
def train_3dunet(checkpoints_dir, training_loader_patches, device, num_epochs=101):

    if os.path.exists(checkpoints_dir) == False:
        os.makedirs(checkpoints_dir)

    # Open the log file
    log_file = open(os.path.join(checkpoints_dir,"train_ssim-l1_log.txt"), "w")

    # Define the generator 
    generator, g_optimizer = get_3dunet_model_and_optimizer(device)

    # Define the learning rate scheduler
    scheduler = MultiStepLR(g_optimizer, milestones=[5, 15, 35], gamma=0.1)

    # Training loop
    for epoch in range(num_epochs):
        for i, batch in enumerate(training_loader_patches):
            # Generate high resolution images
            recon, hifi = prepare_batch(batch, device)
            generated_images = generator(recon)

            # Train the generator
            g_optimizer.zero_grad()
            g_loss = ssim_l1_loss(generated_images, hifi, device)
            g_loss.backward()
            g_optimizer.step()

            gc.collect()
            torch.cuda.empty_cache()
            
            if (i%100)==0:
                # Print the progress
                message = "Epoch [{}/{}], Mix-loss: {:.4f}, Learning Rate: {}".format(epoch+1, 
                                                                                      num_epochs, 
                                                                                      g_loss.item(), 
                                                                                      g_optimizer.param_groups[0]["lr"])
                print(message)
                log_file.write(message + "\n")
                sys.stdout.flush()
                
        scheduler.step()
        if (epoch%10)==0:
            torch.save(generator.state_dict(), os.path.join(checkpoints_dir,'UNet_combine_weights_'+str(epoch)+'_ssim-l1.pth'))


# if script is run
if __name__ == '__main__':
    # parse command line args
    parser = argparse.ArgumentParser(description='Train 3D patch-based UNet to reconstruct hifi images')
    
    # Input images source
    parser.add_argument('-images_dir','--images_dir',
                        help='Directory where coregistered reshaped reconstructed niftis from each plane are stored',
                        required=True,
                        )
    
    # Hifi and mask source
    parser.add_argument('-hifi_and_mask_dir','--hifi_and_mask_dir',
                        help='Directory where hifi and brainmask images are stored',
                        required=True,
                        )
    
    # output directory
    parser.add_argument('-output_dir','--output_dir',
                        help='Directory to output the weights of the UNet after each epoch (checkpoints_dir)',
                        required=True,
                        )
    
    # (OPTIONAL) Number of training epochs
    parser.add_argument('-num_epochs','--num_epochs',
                        help='Number of epochs during training',
                        default=101,
                        required=False,
                        )
    
    # (OPTIONAL) training batch size
    parser.add_argument('-training_batch_size','--training_batch_size',
                        help='Batch size during training',
                        default=8,
                        required=False,
                        )
    
    # (OPTIONAL) max queue length
    parser.add_argument('-max_queue_length','--max_queue_length',
                        help='Maximum length of queue before refilling',
                        default=300,
                        required=False,
                        )

    # (OPTIONAL) Samples per volume
    parser.add_argument('-samples_per_volume','--samples_per_volume',
                        help='Number of patches sampled from each volume during training',
                        default=80,
                        required=False,
                        )
    
    # (OPTIONAL) Number of processes
    parser.add_argument('-num_workers','--num_workers',
                        help='Batch size during training',
                        default=multiprocessing.cpu_count(),
                        required=False,
                        )
    
    # (OPTIONAL) Shuffle subjects
    parser.add_argument('-shuffle_subjects','--shuffle_subjects',
                        help='Shuffle subjects during training',
                        default=True,
                        required=False,
                        )
    
    # (OPTIONAL) Shuffle patches
    parser.add_argument('-shuffle_patches','--shuffle_patches',
                        help='Shuffle patches during training',
                        default=True,
                        required=False,
                        )

    args = parser.parse_args()

    print(args)

    print('Starting')

    # device
    device = torch.device('cuda') if torch.cuda.is_available() else 'cpu'

    # set seed
    seed = 17  # for reproducibility
    random.seed(seed)
    torch.manual_seed(seed)

    # create training patches
    training_patches = create_patches_training_set(
        images_dir=Path(args.images_dir),
        hifi_and_mask_dir=Path(args.hifi_and_mask_dir),
        training_batch_size=int(args.training_batch_size),
        max_queue_length=int(args.max_queue_length),
        samples_per_volume=int(args.samples_per_volume),
        sampler=tio.data.WeightedSampler((32,32,32), 'sampling_map'),
        num_workers=int(args.num_workers),
        shuffle_subjects=bool(args.shuffle_subjects),
        shuffle_patches=bool(args.shuffle_patches)
    )

    print('Training patches created')

    print('Starting training')

    # train the UNet
    train_3dunet(
        checkpoints_dir=args.output_dir,
        training_loader_patches=training_patches,
        device=device,
        num_epochs=int(args.num_epochs)
    )

    print('Finished training')


