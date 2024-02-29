import numpy as np
import os
import ants
import nibabel as nib
import argparse
import multiprocessing
import math
import sys

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

    return make_cube(vol3D)

# pad subject images
def pad_subject_images(data_source, subject, output_dir):
    print(f'Processing: {subject}')

    t1 = nib.load(os.path.join(data_source, subject, 'derivatives', 'registered_images', f'{subject}_hifi_T1_skullstripped.nii.gz'))
    t2 = nib.load(os.path.join(data_source, subject, 'derivatives', 'registered_images', f'{subject}_hifi_T2_skullstripped.nii.gz'))
    flair = nib.load(os.path.join(data_source, subject, 'derivatives', 'registered_images', f'{subject}_hifi_FLAIR_skullstripped.nii.gz'))

    # copy headers and affines
    t1_header, t1_affine = t1.header.copy(), t1.affine.copy()
    t2_header, t2_affine = t2.header.copy(), t2.affine.copy()
    flair_header, flair_affine = flair.header.copy(), flair.affine.copy()

    # normalize and convert to cube
    t1 = norm_zero_to_one(t1.get_fdata())
    t2 = norm_zero_to_one(t2.get_fdata())
    flair = norm_zero_to_one(flair.get_fdata())
    brainmask = (t1>0.10).astype(int)*50 + 1
    
    print(t1.shape)
    print(t2.shape)
    print(flair.shape)
    print(brainmask.shape)

    # create images
    t1_image = nib.Nifti1Image(t1, t1_affine, t1_header)
    t2_image = nib.Nifti1Image(t2, t2_affine, t2_header)
    flair_image = nib.Nifti1Image(flair, flair_affine, flair_header)
    brainmask_image = nib.Nifti1Image(brainmask, t1_affine, dtype='int64')

    # t1 output
    t1_filepath = os.path.join(output_dir, f'{subject}_hifi_t1_padded.nii.gz')

    # t2 output
    t2_filepath = os.path.join(output_dir, f'{subject}_hifi_t2_padded.nii.gz')

    # flair output
    flair_filepath = os.path.join(output_dir, f'{subject}_hifi_flair_padded.nii.gz')

    # brainmask output
    brainmask_filepath = os.path.join(output_dir, f'{subject}_hifi_brainmask.nii.gz')

    # save images
    nib.save(t1_image, t1_filepath)
    nib.save(t2_image, t2_filepath)
    nib.save(flair_image, flair_filepath)
    nib.save(brainmask_image, brainmask_filepath)

    save_success = os.path.exists(t1_filepath) and os.path.exists(t2_filepath) and os.path.exists(flair_filepath) and os.path.exists(brainmask_filepath)

    return save_success

# register single image to target
def register_image(path_source, path_target):

    # Load the source and target images
    source_image = ants.image_read(path_source)
    target_image = ants.image_read(path_target)

    # Perform image registration
    registration_result = ants.registration(
        fixed=target_image,
        moving=source_image,
        type_of_transform="DenseRigid",
        verbose=False
    )

    # Get the registered image
    registered_image = registration_result['warpedmovout']
    
    return registered_image

# save registered image
def save_registered_image(path_source, path_target, out_path):

    registered_image = register_image(path_source, path_target)

    ants.image_write(registered_image, out_path)

    success = os.path.exists(out_path)

    return success

# coregister images for a single subject in a single modality
def coregister_hifi_t2_flair_single_subject(subject, modality, hifi_dir, out_dir):
    # name of hifi t1 padded image
    reshaped_hifi_t1_name = f'{subject}_hifi_t1_padded.nii.gz'

    # path to image
    reshaped_hifi_t1_path = os.path.join(hifi_dir, reshaped_hifi_t1_name)

    # output path
    registered_hifi_t1_path = os.path.join(out_dir, reshaped_hifi_t1_name)

    # copy hifi t1 to output directory
    os.system(f'cp {reshaped_hifi_t1_path} {registered_hifi_t1_path}')

    # name of hifi brainmask image
    reshaped_hifi_brainmask_name = f'{subject}_hifi_brainmask.nii.gz'

    # path to image
    reshaped_hifi_brainmask_path = os.path.join(hifi_dir, reshaped_hifi_brainmask_name)

    # output path
    registered_hifi_brainmask_path = os.path.join(out_dir, reshaped_hifi_brainmask_name)

    # copy hifi brainmask to output directory
    os.system(f'cp {reshaped_hifi_brainmask_path} {registered_hifi_brainmask_path}')

    # name of hifi t2 or flair image
    reshaped_image_name = f'{subject}_hifi_{modality}_padded.nii.gz'

    # path to image
    reshaped_image_path = os.path.join(hifi_dir, reshaped_image_name)

    print(f'Processing {subject} hifi {modality}')

    # output path
    registered_image_path = os.path.join(out_dir, reshaped_image_name)

    # coregister hifi t2 or flair image to hifi t1
    successful_registration = save_registered_image(reshaped_image_path, reshaped_hifi_t1_path, registered_image_path)

    if successful_registration == True:
        print(f'Successfully registered {subject} hifi {modality}')
    
    else:
        print(f'Failed registration of {subject} hifi {modality}')
        sys.exit()

# coregister images for a single subject in a single modality
def coregister_axial_coronal_sagittal_single_subject(subject, plane, modality, hifi_dir, reshape_dir, out_dir):
    # name of hifi t1 padded image
    reshaped_hifi_t1_name = f'{subject}_hifi_t1_padded.nii.gz'

    # path to image
    reshaped_hifi_t1_path = os.path.join(hifi_dir, reshaped_hifi_t1_name)

    # name of axial, coronal, or sagittal image
    reshaped_image_name = f'{subject}_recon_{modality}_{plane}.nii.gz'

    # path to image
    reshaped_image_path = os.path.join(reshape_dir, reshaped_image_name)

    print(f'Processing {subject} {plane} {modality}')

    # output path
    registered_image_path = os.path.join(out_dir, reshaped_image_name)

    # coregister axial, coronal, or sagittal image to hifi t1
    successful_registration = save_registered_image(reshaped_image_path, reshaped_hifi_t1_path, registered_image_path)

    if successful_registration == True:
        print(f'Successfully registered {subject} {plane} {modality}')
    
    else:
        print(f'Failed registration of {subject} {plane} {modality}')
        sys.exit()

# iterate padding over a list of subjects
def iterate_padding_for_each_sub(data_source, subjects_list, output_dir):

    # iterate for each subject
    for sub in subjects_list:
        success = pad_subject_images(data_source, sub, output_dir)

        if success == True:
            print(f'Completed: {sub}')
        
        else:
            print(f'Failed: {sub}')

# coregister all images for a single subject
def coregister_all_images_single_subject(subject, hifi_dir, reshape_dir, out_dir):

    hifi_modalities = ['t2', 'flair']

    for hifi_modality in hifi_modalities:
        coregister_hifi_t2_flair_single_subject(subject, hifi_modality, hifi_dir, out_dir)
    
    modalities = ['t1', 't2', 'flair']
    planes = ['axial', 'coronal', 'sagittal']

    for plane in planes:
        for modality in modalities:
            coregister_axial_coronal_sagittal_single_subject(subject, plane, modality, hifi_dir, reshape_dir, out_dir)

# iterate for each fold
def iterate_for_each_fold(data_source, output_dir, n_splits=12):
    # get list of folds
    folds = [f'fold_{fold}' for fold in range(0, int(n_splits))]

    for fold in folds:
        # all image outputs in axial plane
        all_images = os.listdir(os.path.join(output_dir, fold, 'recon_niftis_training_set_reshaped'))

        full_subject_list = []

        for image in all_images:
            subject = image.split('_')[0]
            full_subject_list.append(subject)
        
        # remove duplicates
        full_subject_list = list(set(full_subject_list))

        output_dir_fold_padding = os.path.join(output_dir, fold, 'hifi_and_brainmask_padded')

        if os.path.exists(output_dir_fold_padding) == False:
            os.makedirs(output_dir_fold_padding)

        # pad all images for each subject
        iterate_padding_for_each_sub(data_source, full_subject_list, output_dir_fold_padding)

        # reshaped training images for fold
        reshape_dir_fold = os.path.join(output_dir, fold, 'recon_niftis_training_set_reshaped')

        # output dir for fold for coregistration
        out_dir_fold_coregistration = os.path.join(output_dir, fold, '3d_unet_training_data')

        if os.path.exists(out_dir_fold_coregistration) == False:
            os.makedirs(out_dir_fold_coregistration)
    
        for sub in full_subject_list:
            coregister_all_images_single_subject(sub, output_dir_fold_padding, reshape_dir_fold, out_dir_fold_coregistration)

# if script is actually run
if __name__ == '__main__':
    # parse command line args
    parser = argparse.ArgumentParser(description='Pad hifi images and brainmask')

    # data source
    parser.add_argument('-data','--data',
                        help='Directory with original coregistered images',
                        required=True,
                        )
    
    # output directory
    parser.add_argument('-output_dir','--output_dir',
                        help='Directory to output the images',
                        required=True,
                        )
    
    # (OPTIONAL) Number of folds
    parser.add_argument('-n_splits', '--n_splits',
                        help='Number of folds',
                        default=12,
                        required=False,
                        )
    
    # (OPTIONAL) process in parallel
    parser.add_argument('-parallel', '--parallel',
                        help='Run reconstruction in parallel',
                        default=False,
                        required=False,
                        )

    args = parser.parse_args()

    print(args)

    print('Starting')

    # process in parallel, 5 at a time
    if bool(args.parallel) == True:
        max_processes_padding = 5

        max_processes_coregistration = 2

        list_of_arguments_padding = []

        list_of_arguments_coregistration = []

        # get list of folds
        folds = [f'fold_{fold}' for fold in range(0, int(args.n_splits))]

        for fold in folds:
            # all image outputs in axial plane
            all_images = os.listdir(os.path.join(args.output_dir, fold, 'recon_niftis_training_set_reshaped'))

            full_subject_list = []

            for image in all_images:
                subject = image.split('_')[0]
                full_subject_list.append(subject)
            
            # remove duplicates
            full_subject_list = list(set(full_subject_list))

            output_dir_fold_padding = os.path.join(args.output_dir, fold, 'hifi_and_brainmask_padded')

            if os.path.exists(output_dir_fold_padding) == False:
                os.makedirs(output_dir_fold_padding)

            list_of_arguments_padding.append((args.data, full_subject_list, output_dir_fold_padding))

            # reshaped training images for fold
            reshape_dir_fold = os.path.join(args.output_dir, fold, 'recon_niftis_training_set_reshaped')

            # output dir for fold for coregistration
            out_dir_fold_coregistration = os.path.join(args.output_dir, fold, '3d_unet_training_data')

            if os.path.exists(out_dir_fold_coregistration) == False:
                os.makedirs(out_dir_fold_coregistration)

            for sub in full_subject_list:
                list_of_arguments_coregistration.append((sub, output_dir_fold_padding, reshape_dir_fold, out_dir_fold_coregistration))
        
        # spawn up 5 processes at once
        with multiprocessing.Pool(processes=max_processes_padding) as pool:
            pool.starmap(iterate_padding_for_each_sub, list_of_arguments_padding)
        
        print('Finished parallel padding')

        # spawn up 2 processes at once
        with multiprocessing.Pool(processes=max_processes_coregistration) as pool:
            pool.starmap(coregister_all_images_single_subject, list_of_arguments_coregistration)

        print('Finished parallel coregistration')

    # run in series
    else:
        iterate_for_each_fold(
            data_source=args.data,
            output_dir=args.output_dir,
            n_splits=int(args.n_splits)
        )

    print('Finished')


