import numpy as np
import glob
import os
import shutil
import argparse
import multiprocessing
from sklearn.model_selection import KFold

# create k folds
def create_k_folds(subject_list, n_splits, shuffle=True):
    kf = KFold(n_splits=n_splits, random_state=1, shuffle=shuffle)
    kf.get_n_splits(subject_list)

    list_of_splits = []

    for i, (train_index, test_index) in enumerate(kf.split(subject_list)):
        list_of_splits.append((train_index, test_index))
    
    return list_of_splits

# get subjects in each split
def get_subjects_in_split(list_of_splits, subject_list, n_splits):
    subjects_in_splits = []

    for i in range(n_splits):
        train_index = list_of_splits[i][0]
        test_index = list_of_splits[i][1]

        subjects_in_splits.append(([subject_list[j] for j in train_index], [subject_list[k] for k in test_index]))

    return subjects_in_splits

# make directory for fold
def make_fold_dir(input_dir, fold_number):
    fold_path = os.path.join(input_dir, f'fold_{fold_number}')

    if os.path.exists == False:
        os.makedirs(fold_path)
    
    print(fold_path)

    return fold_path

# make directory for each plane for fold
def make_plane_dirs(input_dir, fold_number):
    # path to fold directory
    fold_path = make_fold_dir(input_dir, fold_number)

    # create the new pix2pix dataset folder for each plane
    planes = ['axial', 'coronal', 'sagittal']

    plane_paths = []

    for plane in planes:
        # path of directory for that plane
        plane_path = os.path.join(fold_path, f'dataset_{plane}_pix2pix')

        plane_paths.append(plane_path)

        # create plane directory
        if os.path.exists(plane_path) == False:
            os.makedirs(plane_path)
    
    print(plane_paths)

    return plane_paths

# make train and test directories in pix2pix format
def make_train_test_dir(data_source, input_dir, fold_number, train_list, test_list):

    # path to fold directory
    fold_path = make_fold_dir(input_dir, fold_number)

    # create the new pix2pix dataset folder for each plane
    planes = ['axial', 'coronal', 'sagittal']

    plane_paths = make_plane_dirs(input_dir, fold_number)

    # make train and test directories
    for plane in planes:
        print(f'Copying files for Fold {fold_number} plane {plane}...')
        plane_path = os.path.join(fold_path, f'dataset_{plane}_pix2pix')

        # make train dir hifi
        train_dir = os.path.join(plane_path, 'train')

        if os.path.exists(train_dir) == False:
            os.makedirs(train_dir)
        
        # make test dir hifi
        test_dir = os.path.join(plane_path, 'test')

        if os.path.exists(test_dir) == False:
            os.makedirs(test_dir)
        
        # copy files for train directory
        for sub in train_list:
            # all images for subject in plane
            sub_files = glob.glob(os.path.join(data_source, f'dataset_{plane}_pix2pix', sub+'*'))

            # copy images
            for png_file in sub_files:
                shutil.copyfile(png_file, os.path.join(train_dir, png_file.split('/')[-1]))
        
        # copy files for test directory
        for sub in test_list:
            # all images for subject in plane
            sub_files = glob.glob(os.path.join(data_source, f'dataset_{plane}_pix2pix', sub+'*'))

            # copy images
            for png_file in sub_files:
                shutil.copyfile(png_file, os.path.join(test_dir, png_file.split('/')[-1]))

# split data for each fold
def iterate_for_each_fold(data_source, input_dir, subs_file, n_splits=12, shuffle=True):
    # file containing subjects
    subject_file = open(subs_file, 'r')

    # list of all subjects
    full_subject_list = []

    # convert to list
    for line in subject_file:
        line = line.split('\n')[0]
        full_subject_list.append(line)
    
    # folds
    folds = get_subjects_in_split(create_k_folds(full_subject_list, n_splits=n_splits, shuffle=shuffle), 
                                  full_subject_list, n_splits)
    
    # make fold dir
    for fold in range(n_splits):
        print(f'Fold {fold}')
        # make the fold directory
        make_fold_dir(input_dir, fold)

    # make directories for each plane
    for fold in range(n_splits):
        print(f'Fold {fold}')
        # make the fold directory
        make_plane_dirs(input_dir, fold)
    
    # iterate for each fold
    for fold in range(n_splits):
        print(f'Fold {fold}')

        train_list = folds[fold][0]
        test_list = folds[fold][1]

        # copy images
        make_train_test_dir(data_source, input_dir, fold, train_list, test_list)


if __name__ == "__main__":
    # parse command line args
    parser = argparse.ArgumentParser(description='Split images into train and test in pix2pix format for each fold')
    
    # data source
    parser.add_argument('-data','--data',
                        help='Directory with data in pix2pix format',
                        required=True,
                        )
    
    # input directory
    parser.add_argument('-input_dir','--input_dir',
                        help='Directory where the inputs for the model (images) will go',
                        required=True,
                        )
    
    # file with all subjects
    parser.add_argument('-subs_file','--subs_file',
                        help='File with all of the subjects',
                        required=True,
                        )
    
    # (OPTIONAL) number of folds
    parser.add_argument('-n_splits', '--n_splits',
                        help='Number of folds',
                        required=False,
                        default=12,
                        )
    
    # (OPTIONAL) shuffle subjects
    parser.add_argument('-shuffle', '--shuffle',
                        help='Shuffle subjects',
                        required=False,
                        default=True,
                        )
    
    # (OPTIONAL) run in parallel
    parser.add_argument('-parallel', '--parallel',
                        help='Run in parallel',
                        required=False,
                        default=False,
                        )
    
    args = parser.parse_args()

    # print arguments
    print(args)

    print('Starting')

    if bool(args.parallel) == True:
        print('Processing in parallel')
        
        max_processes = 12

        # file containing subjects
        subject_file = open(args.subs_file, 'r')

        # list of all subjects
        full_subject_list = []

        # convert to list
        for line in subject_file:
            line = line.split('\n')[0]
            full_subject_list.append(line)
        
        # folds
        folds = get_subjects_in_split(create_k_folds(full_subject_list, n_splits=int(args.n_splits), shuffle=bool(args.shuffle)), 
                                      full_subject_list, int(args.n_splits))
        
        # make fold dir
        for fold in range(int(args.n_splits)):
            print(f'Fold {fold}')
            # make the fold directory
            make_fold_dir(args.input_dir, fold)

        # make directories for each plane
        for fold in range(int(args.n_splits)):
            print(f'Fold {fold}')
            # make the fold directory
            make_plane_dirs(args.input_dir, fold)
        
        list_of_arguments = []

        # iterate for each fold
        for fold in range(int(args.n_splits)):
            print(f'Fold {fold}')

            train_list = folds[fold][0]
            test_list = folds[fold][1]

            list_of_arguments.append((args.data, args.input_dir, fold, train_list, test_list))
        
        with multiprocessing.Pool(processes=max_processes) as pool:
            pool.starmap(make_train_test_dir, list_of_arguments)
        
        print('Finished parallel processing')
    
    else:
        iterate_for_each_fold(
            data_source=args.data,
            input_dir=args.input_dir,
            subs_file=args.subs_file,
            n_splits=int(args.n_splits),
            shuffle=bool(args.shuffle),
        )
    
    print('Finished')
    

