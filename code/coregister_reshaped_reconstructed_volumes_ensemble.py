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
def coregister_t1_t2_flair_single_subject(subject, modality, reshape_dir, out_dir):
    if os.path.exists(out_dir) == False:
        os.makedirs(out_dir)

    planes = ['coronal', 'sagittal']

    reshaped_axial_name = f'{subject}_recon_{modality}_axial.nii.gz'

    reshaped_axial_path = os.path.join(reshape_dir, reshaped_axial_name)

    registered_axial_path = os.path.join(out_dir, reshaped_axial_name)

    os.system(f'cp {reshaped_axial_path} {registered_axial_path}')

    for plane in planes:
        reshaped_image_name = f'{subject}_recon_{modality}_{plane}.nii.gz'

        reshaped_image_path = os.path.join(reshape_dir, reshaped_image_name)

        print(f'Processing {subject} {plane} {modality}')

        registered_image_path = os.path.join(out_dir, reshaped_image_name)

        successful_registration = save_registered_image(reshaped_image_path, reshaped_axial_path, registered_image_path)

        if successful_registration == True:
            print(f'Successfully registered {subject} {plane} {modality}')
        
        else:
            print(f'Failed registration of {subject} {plane} {modality}')
            sys.exit()

# iterate over each modality
def coregister_all_modalities_single_subject(subject, reshape_dir, out_dir):
    modalities = ['t1', 't2', 'flair']

    for modality in modalities:
        coregister_t1_t2_flair_single_subject(subject, modality, reshape_dir, out_dir)

# iterate over list of subjects
def coregister_all_subjects(subjects_list, reshape_dir, out_dir): 
    for sub in subjects_list:
        coregister_all_modalities_single_subject(sub, reshape_dir, out_dir)
    
# iterate over fold
def coregister_all_folds(subs_file, data_dir, out_dir, n_splits=12):
    # get list of folds
    folds = [f'fold_{fold}' for fold in range(0, int(n_splits))]

    full_subject_list = get_subject_list(subs_file)

    for fold in folds:
        # all image outputs in axial plane
        all_images = os.listdir(os.path.join(data_dir, fold, 'recon_niftis_reshaped'))

        reshaped_images_dir = os.path.join(data_dir, fold, 'recon_niftis_reshaped')
        output_dir = os.path.join(out_dir, fold, 'recon_niftis_reshaped_coregistered')

        coregister_all_subjects(full_subject_list, reshaped_images_dir, output_dir)



# if script is run
if __name__ == '__main__':
    # parse command line args
    parser = argparse.ArgumentParser(description='Coregister reshaped nifti volumes from ensemble model')

    # subs file
    parser.add_argument('-subs_file','--subs_file',
                        help='File with list of subjects',
                        required=True,
                        )
    
    # data source
    parser.add_argument('-data','--data',
                        help='Directory with where reshaped reconstructed niftis from each plane are stored',
                        required=True,
                        )
    
    # output directory
    parser.add_argument('-output_dir','--output_dir',
                        help='Directory to output the coregistered reshaped niftis',
                        required=True,
                        )
    
    # number of folds
    parser.add_argument('-n_splits','--n_splits',
                        help='Number of folds',
                        required=False,
                        default=12,
                        )

    args = parser.parse_args()

    print(args)

    print('Starting')

    coregister_all_folds(
        subs_file=os.path.abspath(args.subs_file),
        data_dir=os.path.abspath(args.data),
        out_dir=os.path.abspath(args.output_dir),
        n_splits=int(args.n_splits)
    )
    
    print('Finished')

