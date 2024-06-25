import os
import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib
from tqdm import tqdm
import ants
import sys
import argparse
import multiprocessing

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

# resample image to have isotropic voxels using the smallest dimension
def resample(img_path, out_path):
    img_ants = ants.image_read(img_path)
    min_dim = np.min(img_ants.shape)
    img_ants_resampled = ants.resample_image(img_ants,[min_dim,min_dim,min_dim], use_voxels=True)
    ants.image_write(img_ants_resampled,out_path)

# reshape image constructed from axial slices
def fix_axial(img_path, out_path):
    axial = nib.load(img_path)
    print(axial.shape)
    min_dim = np.min(axial.shape)
    max_dim = np.max(axial.shape)

    if max_dim <= 256:
        axial.header['dim'] = [3, 256, 256, min_dim, 1, 1, 1, 1]
        axial.header['pixdim'] = [1,1,1,256/min_dim,1,1,1,1]

        lr_flip = nib.Nifti1Image(np.swapaxes(axial.get_fdata(),0,2), axial.affine, axial.header)
        lr_flip.affine[1,1] = -1
        lr_flip.affine[2,2] = -1 * (256/min_dim)
    else:
        axial.header['dim'] = [3, 256, 256, max_dim, 1, 1, 1, 1]
        axial.header['pixdim'] = [1,1,1,256/max_dim,1,1,1,1]

        lr_flip = nib.Nifti1Image(np.swapaxes(axial.get_fdata(),0,2), axial.affine, axial.header)
        lr_flip.affine[1,1] = -1
        lr_flip.affine[2,2] = -1 * (256/max_dim)
    nib.save(lr_flip,out_path)

    # resample reshaped image
    resample(out_path, out_path)

# reshape image constructed from coronal slices
def fix_coronal(img_path, out_path):
    coronal = nib.load(img_path)
    print(coronal.shape)
    min_dim = np.min(coronal.shape)
    max_dim = np.max(coronal.shape)

    lr_flip = nib.Nifti1Image(np.swapaxes(np.swapaxes(coronal.get_fdata(),0,2),1,2), coronal.affine, coronal.header)
    if max_dim <= 256:
        lr_flip.affine[1,1] = -1*(256/min_dim)
    else:
        lr_flip.affine[1,1] = -1*(256/max_dim)
    lr_flip.affine[2,2] = -1

    nib.save(lr_flip,out_path)

    # resample reshaped image
    resample(out_path, out_path)

# reshape image constructed from sagittal slices
def fix_sagittal(img_path, out_path):
    sagittal = nib.load(img_path)
    print(sagittal.shape)
    min_dim = np.min(sagittal.shape)
    max_dim = np.max(sagittal.shape)
    new_header = sagittal.header.copy()
    new_affine = sagittal.affine.copy()
    if max_dim <= 256:
        new_affine[0,0] = new_affine[0,0]*(256/min_dim)
    else:
        new_affine[0,0] = new_affine[0,0]*(256/max_dim)
    new_affine[1,1] = -1
    new_affine[2,2] = -1

    lr_flip = nib.Nifti1Image(sagittal.get_fdata(), new_affine, new_header)
    nib.save(lr_flip,out_path)

    # resample reshaped image
    resample(out_path, out_path)

# reshape single sequence images for single subject in a single plane
def reshape_single_sequence(subject, plane, modality, recon_dir, out_dir):
    # name of image
    image_name = f'{subject}_recon_{modality}.nii.gz'

    # name of reconstruction directory for plane
    plane_dir = f'recon_{plane}'

    # path to image
    image_path = os.path.join(recon_dir, plane_dir, image_name)
    
    # name of output file
    out_file = f'{subject}_recon_{modality}_{plane}.nii.gz'

    # path for file output
    out_path = os.path.join(out_dir, out_file)

    # if axial slices
    if plane == 'axial':
        fix_axial(image_path, out_path)
    
    # if coronal slices
    elif plane == 'coronal':
        fix_coronal(image_path, out_path)
    
    # if sagittal slices
    elif plane == 'sagittal':
        fix_sagittal(image_path, out_path)
    
    else:
        print('Plane must be axial, coronal, or sagittal')
        sys.exit()
    
    success = os.path.exists(out_path)

    return success

# reshape t1, t2, and flair images for single subject in single plane
def reshape_t1_t2_flair(subject, plane, recon_dir, out_dir):
    modalities = ['t1', 't2', 'flair']

    # iterate for modalities
    for modality in modalities:
        success = reshape_single_sequence(subject, plane, modality, recon_dir, out_dir)

        if success == True:
            print(f'Reshaping for {subject}_{modality} in the {plane} plane succeeded')

        else:
            print(f'Reshaping for {subject}_{modality} in the {plane} plane failed')
            sys.exit()

# reshape for a list of subjects in a single plane
def reshape_multiple_subjects(subject_list, plane, recon_dir, out_dir):
    # iterate for each subject
    for sub in subject_list:
        reshape_t1_t2_flair(sub, plane, recon_dir, out_dir)

# reshape for a list of subjects in each plane
def iterate_for_each_plane(subject_list, recon_dir, out_dir):
    planes = ['axial', 'coronal', 'sagittal']

    for plane in planes:
        reshape_multiple_subjects(subject_list, plane, recon_dir, out_dir)

# reshape for each fold
def iterate_for_each_fold(subs_file, data_dir, output_dir, n_splits=12):
    # get list of folds
    folds = [f'fold_{fold}' for fold in range(0, int(n_splits))]

    full_subject_list = get_subject_list(subs_file)

    for fold in folds:
        # all image outputs in axial plane
        all_images = os.listdir(os.path.join(data_dir, fold, 'reconstructed_niftis', 'recon_axial'))

        reconstructed_dir_fold = os.path.join(data_dir, fold, 'reconstructed_niftis')

        output_dir_fold = os.path.join(output_dir, fold, 'recon_niftis_reshaped')

        if os.path.exists(output_dir_fold) == False:
            os.makedirs(output_dir_fold)

        iterate_for_each_plane(full_subject_list, reconstructed_dir_fold, output_dir_fold)

# iterate in parallel
def iterate_in_parallel(subs_file, data_dir, output_dir, n_splits=12):
    print('Processing in parallel')
    max_processes = 12

    list_of_arguments = []

    full_subject_list = get_subject_list(subs_file)

    planes = ['axial', 'coronal', 'sagittal']

    # get list of folds
    folds = [f'fold_{fold}' for fold in range(0, n_splits)]

    for fold in folds:
        # all image outputs in axial plane
        all_images = os.listdir(os.path.join(data_dir, fold, 'reconstructed_niftis', 'recon_axial'))

        reconstructed_dir_fold = os.path.join(data_dir, fold, 'reconstructed_niftis')

        output_dir_fold = os.path.join(output_dir, fold, 'recon_niftis_reshaped')

        if os.path.exists(output_dir_fold) == False:
            os.makedirs(output_dir_fold)

        for plane in planes:
            for sub in full_subject_list:
                list_of_arguments.append((sub, plane, reconstructed_dir_fold, output_dir_fold))
    
    with multiprocessing.Pool(processes=max_processes) as pool:
        pool.starmap(reshape_t1_t2_flair, list_of_arguments)

    print('Finished parallel reshaping')


# if script is run
if __name__ == '__main__':
    # parse command line args
    parser = argparse.ArgumentParser(description='Reshape nifti volumes from reconstructed volumes from ensemble model')

    # subs file
    parser.add_argument('-subs_file','--subs_file',
                        help='File containing list of subjects',
                        required=True,
                        )
    
    # data source
    parser.add_argument('-data','--data',
                        help='Directory where reconstructed niftis from each plane are stored',
                        required=True,
                        )
    
    # output directory
    parser.add_argument('-output_dir','--output_dir',
                        help='Directory to output the reshaped niftis',
                        required=True,
                        )
    
    # (OPTIONAL) Number of folds
    parser.add_argument('-n_splits','--n_splits',
                        help='Number of splits',
                        default=12,
                        required=False,
                        )
    
    # (OPTIONAL) parallel
    parser.add_argument('-parallel','--parallel',
                        help='Run in parallel, 12 at a time',
                        default=False,
                        required=False,
                        )

    args = parser.parse_args()

    print(args)

    print('Starting')

    # run in parallel
    if bool(args.parallel) == True:
        iterate_in_parallel(
            subs_file=os.path.abspath(args.subs_file),
            data_dir=os.path.abspath(args.data),
            output_dir=os.path.abspath(args.output_dir),
            n_splits=int(args.n_splits)
        )

    # run in series
    else:
        iterate_for_each_fold(
            subs_file=os.path.abspath(args.subs_file),
            data_dir=os.path.abspath(args.data),
            output_dir=os.path.abspath(args.output_dir),
            n_splits=int(args.n_splits)
        )
    
    print('Finished')
