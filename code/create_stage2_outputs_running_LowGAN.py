import torch
import torchio as tio
import torch.nn as nn
import torch.nn.functional as F
from pytorch_msssim import ssim
from torch.optim.lr_scheduler import MultiStepLR
from unet import UNet
from tqdm.auto import tqdm
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import multiprocessing
import gc
import random
from pathlib import Path
import glob
import nibabel as nib
import math
import argparse

def get_subject_list(subs_file):
    # file containing subjects
    subject_file = open(subs_file, 'r')

    # list of all subjects
    full_subject_list = []

    # convert to list
    for line in subject_file:
        line = line.split('\n')[0]
        full_subject_list.append(line)

    return full_subject_list

# pad into a cube
def make_cube(img):
    # this function zero pads the image until it becomes a cube
    x,y,z = img.shape
    max_dim = np.max(img.shape)
    
    to_add_x = (max_dim - x) / 2
    to_add_y = (max_dim - y) / 2
    to_add_z = (max_dim -  z) / 2
    
    zero_padded = np.ones((int(x+(to_add_x*2)), int(y+(to_add_y*2)), int(z+(to_add_z*2))))*img[0,0,0]
    
    zero_padded[math.floor(to_add_x):x+math.floor(to_add_x), math.floor(to_add_y):y+math.floor(to_add_y), math.floor(to_add_z):z+math.floor(to_add_z)] = img
    
    return zero_padded

# normalize and pad to a cube
def norm_zero_to_one(vol3D):
    '''Normalize a 3D volume to zero to one range

    Parameters:
      vol3D (3D numpy array): 3D image volume
    '''

    # normalize to 0 to 1 range
    vol3D = vol3D - np.min(vol3D) # set lower bound
    vol3D = vol3D / ( np.max(vol3D) - np.min(vol3D) ) # set upper bound

    return vol3D

# Simple 3D UNet model
def get_3dunet_model(device):
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
    return model

# prepare batch for testing
def prepare_batch(batch_in, device):
  recon = batch_in['recon'][tio.DATA].to(device)
  return recon

# create testing set for single subject for a single modality
def create_testing_set(subject, images_dir, modality, batch_size, patch_size, patch_overlap):
    
    # axial, coronal, sagittal images
    image_paths_coronal = sorted(Path(images_dir).glob(f'{subject}_recon_{modality}_coronal.nii.gz'))
    image_paths_axial = sorted(Path(images_dir).glob(f'{subject}_recon_{modality}_axial.nii.gz'))
    image_paths_sagittal = sorted(Path(images_dir).glob(f'{subject}_recon_{modality}_sagittal.nii.gz'))

    testing_subject_4d = []
    for (coronal_path, axial_path, sagittal_path) in zip(image_paths_coronal, image_paths_axial, image_paths_sagittal):
        
        subject = tio.Subject(recon=tio.ScalarImage([coronal_path,axial_path,sagittal_path]))
        testing_subject_4d.append(subject)

    testing_transform = tio.Compose([tio.ZNormalization(masking_method=tio.ZNormalization.mean), tio.Resample('recon')])

    testing_set = tio.SubjectsDataset(testing_subject_4d, transform=testing_transform)

    print('Testing set:', len(testing_set), 'subjects')

    grid_sampler = tio.inference.GridSampler(testing_set[0], patch_size, patch_overlap)
    
    # load patches
    patch_loader = torch.utils.data.DataLoader(grid_sampler, batch_size=batch_size)

    # aggregate patches
    aggregator = tio.inference.GridAggregator(grid_sampler)

    affine = testing_set[0].recon.affine

    return patch_loader, aggregator, affine

# predict on a single subject
def predict_single_subject_single_modality(weights_path, output_dir, device, subject, images_dir, modality, 
                           batch_size, patch_size, patch_overlap):
    print(f'Predicting {subject} {modality}')

    # create patches, aggregator, and affine for subject
    patch_loader, aggregator, affine = create_testing_set(subject, images_dir, modality, batch_size, patch_size, patch_overlap)

    # # create patches for a single subject
    # patch_loader, aggregator = create_patches_single_subject(subject, batch_size, patch_size, patch_overlap)

    if os.path.exists(output_dir) == False:
        os.makedirs(output_dir)
    
    # generate model
    generator = get_3dunet_model(device)

    # load model weights
    generator.load_state_dict(torch.load(weights_path))

    generator.eval()
    with torch.no_grad():
        for patches_batch in patch_loader:
            recon = prepare_batch(patches_batch, device)
            locations = patches_batch[tio.LOCATION]
            recon_3d = generator(recon)
            aggregator.add_batch(recon_3d, locations)
    
    foreground = aggregator.get_output_tensor()
    prediction = tio.ScalarImage(tensor=foreground, affine=affine)

    prediction.save(os.path.join(output_dir, f'{subject}_recon_{modality}.nii.gz'))

# predict on one subject for all modalities
def predict_single_subject(weights_path, output_dir, device, subject, images_dir, 
                            batch_size, patch_size, patch_overlap):
    modalities = ['t1', 't2', 'flair']

    for modality in modalities:
        predict_single_subject_single_modality(weights_path, output_dir, device, subject, images_dir, modality,
                                                batch_size, patch_size, patch_overlap)

# predict on all subjects
def predict_all_subjects(subject_list, checkpoints_dir, output_dir, device, images_dir,
                            batch_size, patch_size, patch_overlap):
    weights_path = os.path.join(checkpoints_dir, 'LowGAN_stage2', 'UNet_combine_weights_100_ssim-l1.pth')

    if os.path.exists(output_dir) == False:
        os.makedirs(output_dir)

    # iterate for each subject
    for sub in subject_list:
        predict_single_subject(weights_path, output_dir, device, sub, images_dir,
                                batch_size, patch_size, patch_overlap)


# if script is run
if __name__ == '__main__':
    # parse command line args
    parser = argparse.ArgumentParser(description='Test 3D patch-based UNet to reconstruct lofi images')
    
    # file with subjects
    parser.add_argument('-subs_file','--subs_file',
                        help='File containing list of subjects',
                        required=True,
                        )
    
    # checkpoints directory
    parser.add_argument('-checkpoints_dir', '--checkpoints_dir',
                        help='Directory with weights of stage 1 and stage 2 models',
                        required=True,
                        )
    
    # images directory
    parser.add_argument('-data_dir','--data_dir',
                        help='Directory with coregistered stage 1 outputs',
                        required=True,
                        )

    # output directory
    parser.add_argument('-output_dir','--output_dir',
                        help='Directory to output stage 2 results',
                        required=True,
                        )
    
    # (OPTIONAL) testing batch size
    parser.add_argument('-batch_size','--batch_size',
                        help='Batch size during testing',
                        default=4,
                        required=False,
                        )
    
    # (OPTIONAL) Patch size
    parser.add_argument('-patch_size','--patch_size',
                        help='Size of patches',
                        default="32,32,32",
                        required=False,
                        )
    
    # (OPTIONAL) Overlap between patches
    parser.add_argument('-patch_overlap','--patch_overlap',
                        help='Overlap between patches',
                        default="4,4,4",
                        required=False,
                        )

    args = parser.parse_args()

    print(args)

    print('Starting')

    device = torch.device('cuda') if torch.cuda.is_available() else 'cpu'

    # set seed
    seed = 17  # for reproducibility
    random.seed(seed)
    torch.manual_seed(seed)

    full_subject_list = get_subject_list(os.path.abspath(args.subs_file))
    
    predict_all_subjects(
        subject_list=full_subject_list,
        checkpoints_dir=Path(args.checkpoints_dir),
        output_dir=Path(args.output_dir),
        device=device,
        images_dir=Path(args.data_dir),
        batch_size=int(args.batch_size),
        patch_size=tuple(map(int, args.patch_size.split(','))),
        patch_overlap=tuple(map(int, args.patch_overlap.split(',')))
    )
    
    print('Finished')

