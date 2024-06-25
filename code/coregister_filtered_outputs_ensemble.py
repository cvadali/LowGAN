import os
import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib
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

# coregister images for a single subject
def coregister_t1_t2_flair_single_subject(subject, ensemble_output_dir, n_splits=12):
    # get list of folds
    folds = [f'fold_{fold}' for fold in range(0, int(n_splits))]

    # modalities
    modalities = ['t1', 't2', 'flair']

    for fold in folds:
        output_dir_fold = os.path.join(ensemble_output_dir, fold, 'recon_niftis_filtered_coregistered_across_folds')

        if os.path.exists(output_dir_fold) == False:
            os.makedirs(output_dir_fold)

        for modality in modalities:
            print(f'Processing {fold} {subject} {modality}')
            source_path = os.path.join(ensemble_output_dir, fold, 'recon_niftis_reshaped_coregistered_filtered', f'{subject}_recon_{modality}.nii.gz')

            target_path = os.path.join(ensemble_output_dir, 'fold_0', 'recon_niftis_reshaped_coregistered_filtered', f'{subject}_recon_t1.nii.gz')

            path_out = os.path.join(output_dir_fold, f'{subject}_recon_{modality}.nii.gz')

            # coregister all images for a given subject to the T1 image for that subject in fold 0
            if fold == 'fold_0' and modality == 't1':
                os.system(f'cp {source_path} {path_out}')

                if os.path.exists(path_out) == True:
                    print(f'Successfully copied {fold} {subject} {modality}')
                else:
                    print(f'Failed to copy {fold} {subject} {modality}')
            
            # coregister all other images to fold 0 T1 image
            else:
                successful_registration = save_registered_image(source_path, target_path, path_out)

                if successful_registration == True:
                    print(f'Successfully registered {fold} {subject} {modality}')
                else:
                    print(f'Failed registration of {fold} {subject} {modality}')
                    sys.exit()

# iterate over list of subjects
def coregister_all_subjects(subs_file, ensemble_output_dir, n_splits=12):
    # get list of subjects
    full_subject_list = get_subject_list(subs_file)

    for subject in full_subject_list:
        coregister_t1_t2_flair_single_subject(subject, ensemble_output_dir, n_splits)


# if script is run
if __name__ == '__main__':
    # parse command line args
    parser = argparse.ArgumentParser(description='Coregister filtered outputs across all folds in ensemble')

    # subs file
    parser.add_argument('-subs_file','--subs_file',
                        help='File containing list of subjects',
                        required=True,
                        )
    
    # output directory
    parser.add_argument('-output_dir','--output_dir',
                        help='Directory where all ensemble outputs are stored',
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

    coregister_all_subjects(
        subs_file=os.path.abspath(args.subs_file),
        ensemble_output_dir=os.path.abspath(args.output_dir),
        n_splits=int(args.n_splits)
    )
    
    print('Finished')



