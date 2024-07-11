import os
import numpy as np
import matplotlib.pyplot as plt
import ants
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

# create numpy array from generated images in one plane
def get_volume_predicted_single_plane(subject, pix2pix_out_dir, model_name, phase='test', epoch='latest'):
    # all image outputs
    all_images = os.listdir(os.path.join(pix2pix_out_dir, model_name, f'{phase}_{str(epoch)}', 'images'))

    # image outputs for subject
    images_sub = [image for image in all_images if subject in image]

    # find number of images that were output for this subject
    largest_number = 0

    for image in images_sub:
        number = int(image.split(f'{subject}_')[-1].split('_')[0])
        if number > largest_number:
            largest_number = number

    # list to store volume
    dicom = []
    
    # iterate for each image
    for i in range(largest_number+1):
        try:
            # filename stem
            file_stem = f'{subject}_{str(i)}'

            # image file name
            # generated images end in _fake_B.png
            recon_filename = os.path.join(pix2pix_out_dir, model_name, f'{phase}_{str(epoch)}', 'images', f'{file_stem}_fake_B.png')

            # add image to list
            dicom.append(plt.imread(recon_filename))

        except Exception as e:
            print(e)
            continue
    return np.array(dicom)

# save reconstructed nifit image
def save_volume_predicted_single_plane(subject, data_dir, output_dir, model_name_stem, plane, phase='test', epoch='latest'):
    print(f'Processing: {subject}_{plane}')

    # name of model used to generate images
    model_name = f'{model_name_stem}_{plane}'

    # directory where generated images were stored
    pix2pix_out_dir = os.path.join(data_dir, f'results_LowGAN')

    # numpy array containing images
    volume = get_volume_predicted_single_plane(subject, pix2pix_out_dir, model_name, phase, epoch)

    # t1, t2, and flair volumes
    t1 = ants.from_numpy(volume[:,:,:,2])
    t2 = ants.from_numpy(volume[:,:,:,1])
    flair = ants.from_numpy(volume[:,:,:,0])

    # output directory for this plane
    output_dir_plane = os.path.join(output_dir, f'recon_{plane}')
    
    # t1 output
    t1_filepath = os.path.join(output_dir_plane, f'{subject}_recon_t1.nii.gz')

    # t2 output
    t2_filepath = os.path.join(output_dir_plane, f'{subject}_recon_t2.nii.gz')

    # flair output
    flair_filepath = os.path.join(output_dir_plane, f'{subject}_recon_flair.nii.gz')

    ants.image_write(t1, t1_filepath)
    ants.image_write(t2, t2_filepath)
    ants.image_write(flair, flair_filepath)

    save_success = os.path.exists(t1_filepath) and os.path.exists(t2_filepath) and os.path.exists(flair_filepath)

    return save_success

# iterate over a list of subjects
def iterate_for_each_sub(full_subject_list, data_dir, output_dir, model_name_stem, plane, phase='test', epoch='latest'):
    # iterate for each subject
    for sub in full_subject_list:
        success = save_volume_predicted_single_plane(sub, data_dir, output_dir, model_name_stem, plane, phase, epoch)

        if success == True:
            print(f'Completed: {sub}_{plane}')
        
        else:
            print(f'Failed: {sub}_{plane}')

# iterate over each plane
def iterate_for_each_plane(full_subject_list, fold, data_dir, output_dir, model_name_stem, phase='test', epoch='latest'):
    print(f'Processing: {fold}')

    planes = ['axial', 'coronal', 'sagittal']

    data_dir_fold = os.path.join(data_dir, fold)

    output_dir_fold = os.path.join(output_dir, fold, 'reconstructed_niftis')

    if os.path.exists(output_dir_fold) == False:
            os.makedirs(output_dir_fold)
    
    model_name_stem_fold = f'{model_name_stem}_{fold}'

    for plane in planes:
        # output directory for this plane
        output_dir_plane = os.path.join(output_dir_fold, f'recon_{plane}')

        if os.path.exists(output_dir_plane) == False:
            os.makedirs(output_dir_plane)
        
        iterate_for_each_sub(full_subject_list, data_dir_fold, output_dir_fold, model_name_stem_fold, plane, phase, epoch)

# iterate for each fold
def iterate_for_each_fold(full_subject_list, data_dir, output_dir, model_name_stem, phase='test', epoch='latest', n_splits=12):
    # get list of folds
    folds = [f'fold_{fold}' for fold in range(0, int(n_splits))]

    for fold in folds:
        iterate_for_each_plane(full_subject_list, fold, data_dir, output_dir, model_name_stem, phase, epoch)


# if script is actually run
if __name__ == '__main__':
    # parse command line args
    parser = argparse.ArgumentParser(description='Reconstruct nifti volumes from generated images')

    # subs file
    parser.add_argument('-subs_file','--subs_file',
                        help='File containing list of subjects',
                        required=True,
                        )

    # model name stem
    parser.add_argument('-model_name_stem','--model_name_stem',
                        help='Stem of the name of the model',
                        required=True,
                        )
    
    # data source
    parser.add_argument('-data','--data',
                        help='Directory with where generated images in each plane are stored',
                        required=True,
                        )
    
    # output directory
    parser.add_argument('-output_dir','--output_dir',
                        help='Directory to output the reconstructed niftis',
                        required=True,
                        )
    
    # (OPTIONAL) phase (train, val, test)
    parser.add_argument('-phase','--phase',
                        help='Phase (test, train, val)',
                        default='test',
                        required=False,
                        )
    
    # (OPTIONAL) epoch of model used to generate images
    parser.add_argument('-epoch','--epoch',
                        help='Epoch of model used to generate images',
                        default='latest',
                        required=False,
                        )
    
    # (OPTIONAL) Number of folds
    parser.add_argument('-n_splits','--n_splits',
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

    full_subject_list = get_subject_list(os.path.abspath(args.subs_file))

    # process in parallel, each fold separately
    if bool(args.parallel) == True:
        max_processes = int(args.n_splits)

        list_of_arguments = []

        # get list of folds
        folds = [f'fold_{fold}' for fold in range(0, int(args.n_splits))]

        for fold in folds:
            list_of_arguments.append((full_subject_list, fold, os.path.abspath(args.data),
                                      os.path.abspath(args.output_dir),
                                      args.model_name_stem, args.phase, args.epoch))
        
        # create each fold separately
        with multiprocessing.Pool(processes=max_processes) as pool:
            pool.starmap(iterate_for_each_plane, list_of_arguments)

    # run in series
    else:
        iterate_for_each_fold(
            full_subject_list=full_subject_list,
            data_dir=os.path.abspath(args.data),
            output_dir=os.path.abspath(args.output_dir),
            model_name_stem=args.model_name_stem,
            phase=args.phase,
            epoch=args.epoch,
            n_splits=int(args.n_splits)
        )

    print('Finished')


