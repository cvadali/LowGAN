import os
import sys
import numpy as np
import nibabel as nib
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

# average single sequence across folds for single subject
def average_single_sequence(subject, modality, input_dir, output_dir, n_splits=12):
    images = []

    image_data = []

    # get list of folds
    folds = [f'fold_{fold}' for fold in range(0, int(n_splits))]

    for fold in folds:
        # image path
        image_path = os.path.join(input_dir, fold, 'recon_niftis_filtered_coregistered_across_folds', f'{subject}_recon_{modality}.nii.gz')

        image = nib.load(image_path)

        images.append(image)

        image_data.append(image.get_fdata())

    
    # average across folds
    averaged_data = np.zeros_like(np.asarray(image_data[0]))

    for fold in folds:
        fold_number = int(fold.split('_')[1])
        averaged_data += (np.asarray(image_data[fold_number]) / n_splits)

    # averaged image
    averaged_image = nib.Nifti1Image(averaged_data, affine=images[0].affine, header=images[0].header)

    output_path = os.path.join(output_dir, f'{subject}_recon_{modality}.nii.gz')

    # save averaged image
    nib.save(averaged_image, output_path)

    success = os.path.exists(output_path)

    return success


# average t1, t2, and flair for single subject
def average_t1_t2_flair(subject, input_dir, output_dir, n_splits=12):
    print(f'Processing: {subject}')
    modalities = ['t1', 't2', 'flair']

    # iterate over sequences
    for modality in modalities:
        success = average_single_sequence(subject, modality, input_dir, output_dir, n_splits)

        if success == True:
            print(f'Saved averaged {modality} for {subject}')
        
        else:
            print(f'Failed to save average {modality} for {subject}')
            sys.exit()
    
    print(f'Finished: {subject}')

# iterate over list of subjects
def iterate_for_each_sub(subs_file, input_dir, output_dir, n_splits=12):
    full_subject_list = get_subject_list(subs_file)

    if os.path.exists(output_dir) == False:
        os.makedirs(output_dir)
    
    # iterate for each subject
    for sub in full_subject_list:
        print(f'Processing: {sub}')
        average_t1_t2_flair(sub, input_dir, output_dir, n_splits)

# iterate in parallel
def iterate_in_parallel(subs_file, input_dir, output_dir, n_splits=12):
    print('Processing in parallel')

    max_processes = 10

    full_subject_list = get_subject_list(subs_file)

    if os.path.exists(output_dir) == False:
        os.makedirs(output_dir)
    
    list_of_arguments = []

    # create list of argument tuples
    for sub in full_subject_list:
        list_of_arguments.append((sub, input_dir, output_dir, int(n_splits)))
    
    # spawn up to 10 processes at once
    with multiprocessing.Pool(processes=max_processes) as pool:
        pool.starmap(average_t1_t2_flair, list_of_arguments)

    print('Finished parallel processing')



# if script is run
if __name__ == '__main__':
    # parse command line args
    parser = argparse.ArgumentParser(description='Average filtered outputs across all folds in ensemble')

    # file with all subjects
    parser.add_argument('-subs_file','--subs_file',
                        help='File with all of the subjects',
                        required=True,
                        )
    
    # data source
    parser.add_argument('-data','--data',
                        help='Directory where coregistered filtered outputs from each fold are stored',
                        required=True,
                        )
    
    # output directory
    parser.add_argument('-output_dir','--output_dir',
                        help='Directory to output the averaged filtered outputs',
                        required=True,
                        )
    
    # (OPTIONAL) number of folds
    parser.add_argument('-n_splits','--n_splits',
                        help='Number of folds',
                        default=12,
                        required=False,
                        )

    # (OPTIONAL) run in parallel
    parser.add_argument('-parallel','--parallel',
                        help='Run in parallel, 10 at a time',
                        default=False,
                        required=False,
                        )

    args = parser.parse_args()

    print(args)

    print('Starting')

    if bool(args.parallel) == True:
        iterate_in_parallel(
            subs_file=os.path.abspath(args.subs_file),
            input_dir=os.path.abspath(args.data),
            output_dir=os.path.abspath(args.output_dir),
            n_splits=int(args.n_splits)
        )

    else:
        iterate_for_each_sub(
            subs_file=os.path.abspath(args.subs_file),
            input_dir=os.path.abspath(args.data),
            output_dir=os.path.abspath(args.output_dir),
            n_splits=int(args.n_splits)
        )

    print('Completed')



